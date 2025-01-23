import torch
import torch.nn as nn, Tensor

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    # Ensure a and b are tensors
    if isinstance(a, tuple):
        a = a[0]

    if isinstance(b, tuple):
        b = b[0]

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)

    # Ensure a and b have at least 2 dimensions (if 1D, reshape to 2D)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)  # Convert to 2D if a is 1D

    if len(b.shape) == 1:
        b = b.unsqueeze(0)  # Convert to 2D if b is 1D

    a_norm = torch.nn.functional.normalize(a, p=2, dim=2)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=2)

    if len(a_norm.shape) == 3:
        a_norm = a_norm.view(-1, a_norm.size(-1))  # Flatten to 2D
    if len(b_norm.shape) == 3:
        b_norm = b_norm.view(-1, b_norm.size(-1))  # Flatten to 2D

    return torch.mm(a_norm, b_norm.transpose(0, 1))

class HardNegativeNLLLoss:
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(self, q_reps: Tensor, d_reps_pos: Tensor, d_reps_neg: Tensor = None):


        if isinstance(d_reps_pos, tuple):
            d_reps_pos = d_reps_pos[0]

        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        full_d_reps_pos = d_reps_pos
        full_q_reps = q_reps
        full_d_reps_neg = d_reps_neg

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)

        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale

        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)

        loss = self.cross_entropy_loss(scores, labels)
        return loss