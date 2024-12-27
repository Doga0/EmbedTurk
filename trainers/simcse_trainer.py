from transformers import Trainer
import torch
import os
from typing import Optional
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

class SimCSETrainer(Trainer):
  def __init__(self, *args, loss_fn, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_fn = loss_fn

  def compute_loss(self, model, inputs, return_outputs=False):

    features, labels = inputs

    q_reps = self.model(features[0])
    d_reps = self.model(features[1])

    d_reps_neg = None
    if len(features) > 2:
        d_reps_neg = self.model(features[2])

    loss = self.loss_function(q_reps, d_reps, d_reps_neg)

    if return_outputs:
        output = torch.cat(
            [model(row)["sentence_embedding"][:, None] for row in features], dim=1
        )
        return loss, output

    return loss

  def _save(self, output_dir: Optional[str] = None, state_dict=None):
    
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")

    self.model.save(output_dir)

    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
