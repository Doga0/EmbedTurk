import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn as nn

from transformers import (
    TrainingArguments,
    LlamaConfig,
    LlamaConfig,
    LlamaConfig,
    GemmaConfig,
    Qwen2Config,
    BitsAndBytesConfig,
    Trainer
  )

from datasets import load_dataset
from accelerate.logging import get_logger
from accelerate import Accelerator

from peft import LoraConfig, get_peft_model

import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from loss.HardNegativeNLLLoss import HardNegativeNLLLoss
from models.llm2vec.llm2vec import LLM2Vec


logger = get_logger(__name__, log_level="INFO")

max_len = 512
test_ratio = 0.1

model_name = "DogaOytc/llama3-mntp-dnm2"
r = 16
lora_alpha = 32
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_dropout = 0.01

pooling_mode = "mean"
size = 400000

loss_scale = 20

### PATHS ####
output_dir = '/content/drive/MyDrive/EmbedTurk/llama3-Instruct-mntp-simcse/results'
cfg_dir = "/content/drive/MyDrive/EmbedTurk/config/ds_config.config"
log_dir = '/content/drive/MyDrive/EmbedTurk//llama3-Instruct-mntp-simcse/logs'
final_model_dir = "/content/drive/MyDrive/EmbedTurk/llama3-Instruct-mntp-simcse/final_model"
tokenizer_dir = "/content/drive/MyDrive/EmbedTurk/llama3-Instruct-mntp-simcse/vocab/tokenizer"

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

  bnb_config = BitsAndBytesConfig(load_in_8bit=True)

  model = LLM2Vec.from_pretrained(
          model_name,
          enable_bidirectional=True,
          pooling_mode=pooling_mode,
          torch_dtype=torch.bfloat16,
          quantization_config = bnb_config,
          max_length=max_len,
          attn_implementation="eager"
      )

  lora_config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    target_modules=target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type=None
  )

  peft_model = model.model
  peft_model = get_peft_model(peft_model, lora_config)

  print("Trainable Parameters: ")
  peft_model.print_trainable_parameters()

  tokenizer = model.tokenizer


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
          item["negative"], 
          pooling_mode=model.pooling_mode
      )
      for item in batch
    ]    

    result = []

    anchor_inputs = tokenizer(anchor, max_length=max_len, return_attention_mask=True, padding='max_length', truncation=True, return_tensors='pt')
    positive_inputs = tokenizer(positives, max_length=max_len, return_attention_mask=True, padding='max_length', truncation=True, return_tensors='pt')
    negative_inputs = tokenizer(negatives, max_length=max_len, return_attention_mask=True, padding='max_length', truncation=True, return_tensors='pt')

    result.append(anchor_inputs)
    result.append(positive_inputs)
    result.append(negative_inputs)
    
    labels = [0] * len(anchor)
    return result,labels


  raw_datasets = load_dataset("emrecan/all-nli-tr", "triplet", split="train")
  np.random.seed(42)
  random_indices = np.random.choice(len(raw_datasets), size=size, replace=False)
  train_dataset = raw_datasets.select(random_indices)

  kwargs = []
  accelerator = Accelerator(kwargs_handlers=kwargs)


  class SupervisedTrainer(Trainer):
        def __init__(
            self,
            *args,
            collate_fn,
            loss_function=None,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.loss_function = loss_function
            self.collate_fn = collate_fn
            self.train_dataloader = DataLoader(
                train_dataset,
                collate_fn=self.collate_fn,
                shuffle=True,
                batch_size=self._train_batch_size,
                pin_memory=True,
            )

        def get_train_dataloader(self):

            return self.accelerator.prepare(self.train_dataloader)

        def compute_loss(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            **kwargs
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            features, labels = inputs

            q_reps = self.model(**features[0]).last_hidden_state
            d_reps = self.model(**features[1]).last_hidden_state
            d_reps_neg = self.model(**features[2]).last_hidden_state

            loss = self.loss_function(q_reps, d_reps, d_reps_neg)
            return loss


  def load_loss(loss_scale):
    loss_class = HardNegativeNLLLoss
    return loss_class(scale=loss_scale)

  loss_fn = load_loss(loss_scale)

  training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_total_limit=2,
    save_steps=5000,
    fp16=True,
    deepspeed=cfg_dir,
    report_to='tensorboard',
    logging_dir=log_dir,
    logging_steps=100,
  )

  if training_args.gradient_checkpointing:
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

  trainer = SupervisedTrainer(
        model=peft_model,
        args=training_args,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
        loss_function=loss_fn,
    )

  gpu_stats = torch.cuda.get_device_properties(0)
  start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
  print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
  print(f"{start_gpu_memory} GB of memory reserved.\n")

  trainer.train()

  peft_model.save_pretrained(final_model_dir)
  tokenizer.save_pretrained(tokenizer_dir)

if __name__=="__main__":
  main()