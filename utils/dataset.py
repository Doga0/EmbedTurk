import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import Any, Optional, Tuple
from transformers import DataCollatorForLanguageModeling

class EmbedTurkDataset(Dataset):
  def __init__(self, tokenizer, dataset, max_len, stride, special_tokens=None):
    self.tokenizer = tokenizer
    self.dataset = dataset
    self.max_len = max_len
    self.stride = stride

    self.special_tokens = ["<|endoftext|>"] if special_tokens is None else special_tokens

    scaler = StandardScaler()
    num_cols = self.dataset.select_dtypes(include='number').columns
    self.dataset[num_cols] = scaler.fit_transform(self.dataset[num_cols])

    self.text_cols = self.dataset.select_dtypes(include=['object', 'string']).columns
    for col in self.text_cols:
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
    return {
      'input_ids': encoded['input_ids'].squeeze(0),
      'attn_mask': encoded['attn_mask'].squeeze(0)
    }

  def __getitem__(self, idx):
    row = self.dataset.iloc[idx]

    numeric_data = torch.tensor(row[self.num_cols].values, dtype=torch.float32) if not self.num_cols.empty else torch.tensor([])

    text_ids = []
    attn_masks = []
    for col in self.text_cols:
      tokenied = row[col]
      text_ids.append(tokenied["input_ids"])
      attn_masks.append(tokenied["attn_masks"])

    text_ids = torch.stack(text_ids)
    attn_masks = torch.stack(attn_masks)

    return {
        'numeric': numeric_data,
        'text': text_ids,
        'attn_mask': attn_masks
    }

  def __len__(self):
    return len(self.dataset)
  
class DataCollatorForLanguageModelingWithFullMasking(DataCollatorForLanguageModeling):
    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% MASK, 0% random, 0% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels