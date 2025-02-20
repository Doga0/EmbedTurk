import torch

from transformers import (
    TrainingArguments,
    LlamaConfig,
    LlamaConfig,
    LlamaConfig,
    GemmaConfig,
    Qwen2Config,
  )

from datasets import load_dataset
from accelerate.logging import get_logger

from peft import LoraConfig, get_peft_model

import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from trainers.sup_trainer import SupSimCSETrainer
from loss.HardNegativeNLLLoss import HardNegativeNLLLoss
from utils.dataset import SupSimCSEDatasetFromHF
from models.llm2vec.llm2vec import LLM2Vec


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

def prepare_for_tokenization(model, text, pooling_mode="mean"):
    if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        )
        return text
    if model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if model.config._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if model.config._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text


def main():
  set_seed(1234)

  device = "cuda" if torch.cuda.is_available() else "cpu"

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
  tokenizer.to(device)


  def collate_fn(batch):

    anchor, positives, negatives = zip(*batch)

    anchor = [
        prepare_for_tokenization(
            model, 
            item["anchor"], 
            pooling_mode=model.pooling_mode
        ) 
        for item in batch
    ]  

    positives = [
        prepare_for_tokenization(
            model, 
            item["positive"], 
            pooling_mode=model.pooling_mode
        ) 
        for item in batch
    ] 

    negatives = [
      prepare_for_tokenization(
          model, 
          neg, 
          pooling_mode=model.pooling_mode
      )
      for item in batch
      for neg in item["negative"]
    ]    

    result = []

    anchor_inputs = tokenizer(anchor, max_length=max_len, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
    positive_inputs = tokenizer(positives, max_length=max_len, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
    negative_inputs = tokenizer(negatives, max_length=max_len, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')

    result.append(anchor_inputs)
    result.append(positive_inputs)
    result.append(negative_inputs)
    
    labels = [0] * len(anchor)
    return result,labels


  raw_datasets = load_dataset("emrecan/all-nli-tr", "triplet", split="train")
  np.random.seed(42)
  random_indices = np.random.choice(len(raw_datasets), size=size, replace=False)
  train_dataset = raw_datasets.select(random_indices)
  train_dataset = SupSimCSEDatasetFromHF(train_dataset, tokenizer)


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

  trainer = SupSimCSETrainer(
      model=peft_model,
      args=training_args,
      train_dataset=train_dataset,
      data_collator=collate_fn,
      loss_function=loss_fn
  )

  trainer.train()

if __name__=="__main__":
  main()