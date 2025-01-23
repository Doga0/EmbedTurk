from transformers import Trainer
import torch
import torch.nn as nn
import os
from typing import Optional
from accelerate.logging import get_logger
from typing import Any, Dict, Optional, Tuple, Union

logger = get_logger(__name__, log_level="INFO")

class SimCSETrainer(Trainer):
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
    return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
      input_ids = inputs.get("input_ids")
      attention_mask = inputs.get("attention_mask")

      # (batch_size, sequence_length)
      q_reps = self.model(input_ids[0:1], attention_mask[0:1], return_dict=False)  
      d_reps = self.model(input_ids[1:2], attention_mask[1:2], return_dict=False) 

      d_reps_neg = None
      if input_ids.size(0) > 2:
          d_reps_neg = self.model(input_ids[2:3], attention_mask[2:3], return_dict=False) 

      # Compute loss
      loss = self.loss_function(q_reps, d_reps, d_reps_neg)

      if return_outputs:
          output = torch.cat(
              [self.model(input_ids[i:i+1], attention_mask[i:i+1])["sentence_embedding"][:, None]
               for i in range(input_ids.size(0))],  # Loop over the batch dimension
              dim=0,  # Concatenate along the batch dimension
          )
          return loss, output

      return loss

  def _save(self, output_dir: Optional[str] = None, state_dict=None):
    # If we are executing this function, we are the process zero, so we don't check for that.
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")

    self.model.save(output_dir)

    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
