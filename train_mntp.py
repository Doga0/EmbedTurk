import torch
import torch.nn as nn
from torch.utils.data import random_split

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
    DataCollatorForLanguageModeling
  )
import datasets
from accelerate.logging import get_logger

from llm2vec.models.bidirectional_llama import LlamaBiForMNTP
from llm2vec.models.bidirectional_mistral import MistralBiForMNTP

from peft import LoraConfig, get_peft_model

from trainers.mntp_trainer import MNTPTrainer
from utils.dataset import DataCollatorForLanguageModelingWithFullMasking, EmbedTurkDataset
from utils.tokenizer import Tokenize

import pandas as pd
import numpy as np
import random
import os
from typing import Any, Dict, List, Optional, Tuple, Union


logger = get_logger(__name__, log_level="INFO")

dataset_path = "/content/drive/MyDrive/EmbedTurk/dataset/new_dataset.csv"
dataset = pd.read_csv(dataset_path)

max_len = 512
stride = 2
train_ratio = 0.8

qlora=True
model_name = "meta-llama/Meta-Llama-3-8B"
r = 8
lora_alpha = 32
target_modules=["q_proj", "v_proj"]
lora_dropout = 0.01
bias="none"
task_type="CAUSAL_LM"

data_collator_type = "default"
mlm_probability = 0.15
line_by_line = False
pad_to_max_length = False
mask_token_type = "blank"

def get_model_class(config):
  config_class_name = config.__class__.__name__

  if config_class_name == "LlamaConfig":
    return LlamaBiForMNTP
  elif config_class_name == "MistralConfig":
    return MistralBiForMNTP
  else:
    raise ValueError()

def dataloader(
      tokenizer, dataset_path, max_len,
      stride, train_ratio, special_tokens=None
    ):

  dataset = EmbedTurkDataset(tokenizer, dataset_path, max_len, stride, special_tokens, train_ratio=0.8,)

  generator = torch.Generator().manual_seed(42)

  train_size = int(len(dataset) * train_ratio)
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

  print(f"Train Size: {train_size}, Val Size: {val_size}")

  return train_dataset, val_dataset

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

  device = "cuda" if torch.cuda.is_available() else "cpu"

  config = AutoConfig.from_pretrained(model_name)
  model_class = get_model_class(config)
    
  if qlora:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

  base_model = model_class.from_pretrained(
      model_name,
      #cache_dir = cache_dir,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
      output_hidden_states=True,
      quantization_config = bnb_config if qlora else None,
      device=device
  )

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

  if tokenizer.mask_token is None:
    if mask_token_type == "blank":
        tokenizer.mask_token = "_"
    elif mask_token_type == "eos":
        tokenizer.mask_token = tokenizer.eos_token
    elif mask_token_type == "mask":
        tokenizer.add_tokens(["<mask>"])
        tokenizer.mask_token = "<mask>"
    else:
        raise ValueError

  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

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

  pad_to_multiple_of_8 = (
        line_by_line
        and training_args.fp16
        and not pad_to_max_length
    )
  
  data_collator_cls = None
  if data_collator_type == "all_mask":
      data_collator_cls = DataCollatorForLanguageModelingWithFullMasking
  elif data_collator_type == "default":
      data_collator_cls = DataCollatorForLanguageModeling
  else:
      raise ValueError(
          f"data_collator_type {data_collator_type} is not supported."
      )

  data_collator = data_collator_cls(
      tokenizer=tokenizer,
      mlm_probability=mlm_probability,
      pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
  )

  train_dataset, val_dataset = dataloader(
      tokenizer,
      dataset_path,
      max_len,
      stride,
      train_ratio,
      special_tokens=None
    )
  
  trainer = MNTPTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
  )

  trainer.train()

  peft_model.save_pretrained("./final_model")
  tok.save_vocab()

if __name__=="__main__":
   main()