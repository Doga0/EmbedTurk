import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    LlamaPreTrainedModel,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    HfArgumentParser,
    LlamaConfig,
  )
import datasets
from accelerate.logging import get_logger

from llm2vec.models.bidirectional_llama import LlamaBiModel

from peft import LoraConfig, get_peft_model

from trainers.simcse_trainer import SimCSETrainer
from loss.HardNegativeNLLLoss import HardNegativeNLLLoss

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import random
import os
from typing import Any, Dict, List, Optional, Tuple, Union


logger = get_logger(__name__, log_level="INFO")

train_ratio = 0.75

dataset_path = "/content/drive/MyDrive/EmbedTurk/dataset/new_dataset.csv"
dataset = pd.read_csv(dataset_path)
max_len = 512
stride = 2
batch_size = 8
shuffle = True
drop_last = True
num_workers = 8

qlora=True
model_name = "meta-llama/Meta-Llama-3-8B"
r = 8
lora_alpha = 32
target_modules=["q_proj", "v_proj"]
lora_dropout = 0.01
bias="none"
task_type="CAUSAL_LM"

class RecDataset(Dataset):
  def __init__(self, tokenizer, dataset, max_len, stride, special_tokens=None):
    self.tokenizer = tokenizer
    self.dataset = dataset
    self.max_len = max_len
    self.stride = stride

    self.special_tokens = ["<|endoftext|>"] if special_tokens is None else special_tokens

    scaler = StandardScaler()
    num_cols = self.dataset.select_dtypes(include='number').columns
    self.dataset[num_cols] = scaler.fit_transform(self.dataset[num_cols])

    text_cols = self.dataset.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
      self.dataset[col] = self.dataset[col].apply(
                    lambda x: self._tokenize_text(str(x))
      )

  def _tokenize_text(self, text):

    encoded = self.tokenizer.encode(
        text,
        max_length=self.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True,
        return_attention_mask=True
    )
    return encoded['input_ids'].squeeze(0).tolist()

  def __getitem__(self, idx):
    row = self.dataset.iloc[idx]

    numeric_data = row[self.num_cols].values if not self.num_cols.empty else torch.tensor([])

    text_data = [row[col] for col in self.text_cols] if not self.text_cols.empty else []

    return {
        'numeric': torch.tensor(numeric_data, dtype=torch.float32),
        'text': torch.tensor(text_data, dtype=torch.long) if text_data else torch.tensor([])
    }

  def __len__(self):
    return len(self.dataset)
  
class Tokenize(AutoTokenizer):
  def __init__(self, tokenizer, dataset):
    self.tokenizer = tokenizer
    self.dataset = dataset

  def encode(
        self, text, truncation, max_len,
        add_special_tokens, return_attention_mask
      ):
    return self.tokenizer.encode_plus(
            text,
            truncation=truncation,
            max_length=max_len,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            padding='max_length' if max_len else False
        )

  def decode(self, ids):
    return self.tokenizer.decode(ids)

  def tokenize_dataset(self, text, truncation, max_len, add_special_tokens):
    return [
        self.encode(
            text,
            truncation,
            max_len,
            add_special_tokens
        )
        for text in self.dataset
    ]

  def extent_vocab(self, new_tokens):
    new_tokens = set(new_tokens) - set(self.tokenizer.get_vocab().keys())

    if new_tokens:
        num_added = self.tokenizer.add_tokens(list(new_tokens))
        self.vocab_size = len(self.tokenizer)
        return num_added
    return 0

  def save_vocab(self):
    self.tokenizer.save_pretrained("./vocab/final_model")

  @property
  def _vocab_size(self):
      return len(self.tokenizer)

def dataloader(
      tokenizer, dataset_path, max_len,
      stride, batch_size, shuffle,
      drop_last, num_workers, special_tokens=None
    ):

  dataset = RecDataset(tokenizer, dataset_path, max_len, stride, special_tokens)

  dataloader = DataLoader(
      dataset, batch_size=batch_size, shuffle=shuffle,
      drop_last=drop_last, num_workers=num_workers
  )

  return dataloader

# TODO: apply Mistral
class EmbedTurkRec(LlamaPreTrainedModel):
  def __init__(self, model_name_or_path, config, qlora):
    super().__init__(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if qlora:
      bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    self.backbone = LlamaBiModel.from_pretrained(
        model_name_or_path,
        #cache_dir = cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        output_hidden_states=True,
        quantization_config = bnb_config if qlora else None,
        device=device
    )

    self.hidden_size = self.backbone.config.hidden_size
    self.dense = nn.Linear(self.hidden_size, self.hidden_size)
    self.activation = nn.Tanh()

    self.init_weights()

  def forward(self, input_ids, attn_mask=None, token_type_ids=None):
    outputs = self.backbone(
        input=input_ids,
        attention_mask=attn_mask,
        token_type_ids=token_type_ids
    )

    cls = outputs.last_hidden_state[:, 0]

    if self.training:
      proj_emb = self.dense(cls)
      emb = self.activation(proj_emb)

    return emb
  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_model(model_name, 
               r, 
               lora_alpha, 
               lora_dropout, 
               qlora=True, 
               target_modules=["q_proj", "v_proj"], 
               bias="none", 
               task_type="CAUSAL_LM"
    ):

  config = LlamaConfig.from_pretrained(model_name)
  base_model = EmbedTurkRec(model_name, config, qlora)

  lora_config = LoraConfig(
      r,
      lora_alpha,
      target_modules,
      lora_dropout,
      bias,
      task_type
  )

  peft_model = get_peft_model(base_model, lora_config)

  print("Trainable Parameters: ")
  peft_model.print_trainable_parameters()

  return peft_model

def main():
    set_seed(1234)

    ''' parser = HfArgumentParser(
            (ModelArguments, TrainingArguments, CustomArguments)
        )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() '''

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tok = Tokenize(tokenizer, dataset)

    peft_model = init_model(
        model_name,
        r,
        lora_alpha,
        lora_dropout,
        qlora,
        target_modules,
        bias,
        task_type
      )

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=10000,
        save_total_limit=2,
        fp16=True, 
        deepspeed="/content/drive/MyDrive/EmbedTurk/dataset/ds_cfg/deepspeed.config",        
        gradient_checkpointing=True,
    	report_to='wandb' 
    )


    train_size = int(len(dataset) * train_ratio)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    print("Train size: ", len(train_dataset))
    print("Val size: ", len(val_dataset))

    train_loader = dataloader(
        tokenizer,
        train_dataset,
        max_len,
        stride,
        batch_size,
        shuffle,
        drop_last,
        num_workers,
        special_tokens=None
      )

    val_loader = dataloader(
        tokenizer,
        val_dataset,
        max_len,
        stride,
        batch_size,
        shuffle,
        drop_last,
        num_workers,
        special_tokens=None
      )

    loss_fn = HardNegativeNLLLoss()

    trainer = SimCSETrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        loss_fn=loss_fn
      )

    trainer.train()

    peft_model.save_pretrained("./final_model")
    tok.save_vocab()

if __name__=="__main__":
  main()