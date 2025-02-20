import torch
import torch.nn as nn
from transformers import Trainer
from typing import Any, Dict, Optional, Tuple, Union
import os
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class SupSimCSETrainer(Trainer):
  def __init__(
      self,
      *args,
      loss_function=None,
      **kwargs,
  ) -> None:
      super().__init__(*args, **kwargs)
      self.loss_function = loss_function

  def compute_loss(
    self,
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
      input_ids = inputs.get("input_ids")
      attention_mask = inputs.get("attention_mask")

      emb = self.model(input_ids, attention_mask)

      q_reps = emb[0::3]
      d_reps_pos = emb[1::3]
      d_reps_neg = emb[2::3]

      # Compute loss
      loss = self.loss_function(q_reps, d_reps_pos, d_reps_neg)

      return loss

  def _save(self, output_dir: Optional[str] = None, state_dict=None):
    # If we are executing this function, we are the process zero, so we don't check for that.
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")

    self.model.model.save(output_dir)
    self.tokenizer.save_pretrained(output_dir)

    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))