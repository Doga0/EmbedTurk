import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
  )
from datasets import load_dataset
from accelerate.logging import get_logger
import evaluate

from models.llama import LlamaBiForMNTP

from peft import LoraConfig, get_peft_model

import numpy as np

import random
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import chain

from trainers.mntp_trainer import MNTPTrainer
from utils.dataset import DataCollatorForLanguageModelingWithFullMasking

logger = get_logger(__name__, log_level="INFO")

dataset_path = "musabg/wikipedia-tr"

qlora = False
model_name = "unsloth/llama-3-8b-Instruct"

r = 8
lora_alpha = 32
target_modules=["q_proj", "v_proj"]
lora_dropout = 0.01
bias="none"
task_type="CAUSAL_LM"

data_collator_type = "default"
mlm_probability = 0.2
line_by_line = False
pad_to_max_length = False

max_seq_length = 512
size = 50000
test_ratio = 0.2

mask_token_type = "blank"
data_collator_type = "default" #for llama

def get_model_class(config):
  config_class_name = config.__class__.__name__

  if config_class_name == "LlamaConfig":
    return LlamaBiForMNTP
  else:
    raise ValueError()

def dataloader(dataset, test_ratio=0.1):

  test_size = int(len(dataset) * test_ratio)
  train_size = len(dataset) - test_size

  np.random.seed(42)
  indices = np.random.permutation(len(dataset))

  train_indices = indices[:train_size]
  val_indices = indices[train_size:]

  train_dataset = dataset.select(train_indices)
  val_dataset = dataset.select(val_indices)

  print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")

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
               qlora,
               target_modules=["q_proj", "k_proj", "v_proj"],
               bias="none",
               task_type="CAUSAL_LM"
    ):

  config = AutoConfig.from_pretrained(model_name)

  model_class = get_model_class(config)

  #error while using qlora with deepspeed
  if qlora:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

  base_model = model_class.from_pretrained(
      model_name,
      device_map="auto",
      torch_dtype=torch.bfloat16,
      attn_implementation="sdpa",
      output_hidden_states=True,
      quantization_config = bnb_config if qlora else None
  )

  lora_config = LoraConfig(
      r=r,
      lora_alpha=lora_alpha,
      target_modules=target_modules,
      lora_dropout=lora_dropout,
      bias=bias,
      task_type=task_type
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

  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      padding_side="right",
      use_fast=True
  )

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
      output_dir='/content/drive/MyDrive/EmbedTurk/llama3-Instruct-mntp/results',
      overwrite_output_dir=True,
      num_train_epochs=3,
      per_device_train_batch_size=4,
      save_steps=10000,
      save_total_limit=2,
      fp16=True,
      deepspeed="/content/drive/MyDrive/EmbedTurk/dataset/ds_cfg/ds_config.config",
  	  report_to='wandb',
      evaluation_strategy="steps",
      eval_steps=500,
      logging_dir='/content/drive/MyDrive/EmbedTurk/logs',
      logging_steps=100,
      load_best_model_at_end=True
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

  raw_datasets = load_dataset(dataset_path, split="train")

  del_cols = ['id', 'url', 'title']
  raw_datasets = raw_datasets.remove_columns(del_cols)

  np.random.seed(42)
  random_indices = np.random.choice(len(raw_datasets), size=size, replace=False)
  subset_dataset = raw_datasets.select(random_indices)

  padding = "max_length" if pad_to_max_length else False

  def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )

  tokenized_datasets = subset_dataset.map(
      tokenize_function,
      batched=True,
      remove_columns=['text']
  )

  def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // max_seq_length) * max_seq_length

    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        for k, t in concatenated_examples.items()
    }
    return result

  tokenized_datasets = tokenized_datasets.map(
      group_texts,
      batched=True,
  )

  train_dataset, val_dataset = dataloader(tokenized_datasets, test_ratio=test_ratio)

  def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)

  accuracy_metric = evaluate.load("accuracy")
  f1_metric = evaluate.load("f1")

  def compute_metrics(eval_preds):
    preds, labels = eval_preds

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    labels = labels.reshape(-1)
    preds = preds.reshape(-1)

    mask = labels != -100

    labels = labels[mask]
    preds = preds[mask]

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    return {
        "accuracy": accuracy,
            "f1": f1
        }

  trainer = MNTPTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
  )

  trainer.train()

  # peft_model.save_pretrained("/content/drive/MyDrive/EmbedTurk/llama3-Instruct-mntp/final_model")
  # tokenizer.save_pretrained("/content/drive/MyDrive/EmbedTurk/llama3-Instruct-mntp/vocab/tokenizer/")

if __name__=="__main__":
  main()