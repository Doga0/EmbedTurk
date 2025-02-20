import torch

from transformers import (
    TrainingArguments,
  )
from datasets import load_dataset
from accelerate.logging import get_logger

from peft import LoraConfig, get_peft_model

from trainers.simcse_trainer import SimCSETrainer
from loss.HardNegativeNLLLoss import HardNegativeNLLLoss
from utils.dataset import SimCSEDatasetFromHF
from models.llm2vec.llm2vec import LLM2Vec

import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple, Union


logger = get_logger(__name__, log_level="INFO")

max_len = 512
test_ratio = 0.1

model_name = "DogaOytc/llama3-mntp-dnm2"
r = 16
lora_alpha = 32
target_modules=["q_proj", "k_proj","v_proj"]
lora_dropout = 0.01
bias="none"
task_type="CAUSAL_LM"

pooling_mode = "mean"
size = 4000

loss_scale = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
  set_seed(1234)

  model = LLM2Vec.from_pretrained(
          model_name,
          enable_bidirectional=True,
          pooling_mode=pooling_mode,
          torch_dtype=torch.bfloat16,
          attn_implementation="eager",
      )

  lora_config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    target_modules=target_modules,
    lora_dropout=lora_dropout,
    bias=bias,
    task_type=task_type
  )
  peft_model = model.model
  peft_model = get_peft_model(peft_model, lora_config)

  print("Trainable Parameters: ")
  peft_model.print_trainable_parameters()

  tokenizer = model.tokenizer

  # Load dataset
  raw_datasets = load_dataset("parsak/msmarco-tr", "passages", split="train")
  del_cols = ['pid']
  raw_datasets = raw_datasets.remove_columns(del_cols)
  np.random.seed(42)
  random_indices = np.random.choice(len(raw_datasets), size=size, replace=False)
  train_dataset = raw_datasets.select(random_indices)
  train_dataset = SimCSEDatasetFromHF(train_dataset, tokenizer)

  def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

    keys = batch[0].keys()
    collated_batch = {key: torch.stack([example[key] for example in batch]) for key in keys}

    return collated_batch

  # Define loss function
  def load_loss(loss_scale):
    loss_class = HardNegativeNLLLoss
    return loss_class(scale=loss_scale)

  loss_fn = load_loss(loss_scale)

  training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/EmbedTurk/llama3-Instruct-mntp-simcse/results',
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=1000,
    fp16=True,
    deepspeed="/content/drive/MyDrive/EmbedTurk/dataset/ds_cfg/ds_config.config",
    report_to='wandb',
    logging_dir='/content/drive/MyDrive/EmbedTurk/logs',
    logging_steps=100,
  )

  # Initialize trainer
  trainer = SimCSETrainer(
      model=peft_model,
      args=training_args,
      train_dataset=train_dataset,
      data_collator=collate_fn,
      loss_function=loss_fn
  )

  # Train model
  trainer.train()

if __name__=="__main__":
  main()