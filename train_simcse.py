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
  )
import datasets
from accelerate.logging import get_logger

from llm2vec.models.bidirectional_llama import LlamaBiModel

from peft import LoraConfig, get_peft_model

from trainers.simcse_trainer import SimCSETrainer
from loss.HardNegativeNLLLoss import HardNegativeNLLLoss
from utils.dataset import EmbedTurkDataset
from utils.tokenizer import Tokenize
from models.llama import EmbedTurkRecLlama

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

  config = LlamaConfig.from_pretrained(model_name)
  base_model = EmbedTurkRecLlama(model_name, config, qlora)

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

    train_dataset, val_dataset = dataloader(
      tokenizer,
      dataset_path,
      max_len,
      stride,
      train_ratio,
      special_tokens=None
    )

    loss_fn = HardNegativeNLLLoss()

    trainer = SimCSETrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_fn=loss_fn
      )

    trainer.train()

    peft_model.save_pretrained("./final_model")
    tok.save_vocab()

if __name__=="__main__":
  main()