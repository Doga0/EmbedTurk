from transformers import LlamaPreTrainedModel, BitsAndBytesConfig
import torch
import torch.nn as nn
from llm2vec.models.bidirectional_llama import LlamaBiModel

# TODO: apply Mistral
class EmbedTurkRecLlama(LlamaPreTrainedModel):
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