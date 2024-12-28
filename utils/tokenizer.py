from transformers import AutoTokenizer

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