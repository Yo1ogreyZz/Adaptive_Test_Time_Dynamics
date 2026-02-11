import torch
import torch.nn as nn


class LowRankAdapter(nn.Module):
    """
    delta_A = U @ V, shape (d_inner, d_state)
    """

    def __init__(self, d_inner, d_state, rank=8, max_delta_norm=0.5, device=None, dtype=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.rank = rank
        self.max_delta_norm = max_delta_norm

        factory_kwargs = {"device": device, "dtype": dtype}
        self.U = nn.Parameter(torch.zeros(d_inner, rank, **factory_kwargs))
        self.V = nn.Parameter(torch.randn(rank, d_state, **factory_kwargs) * 0.1)

    def forward(self):
        delta = torch.matmul(self.U, self.V)
        if self.max_delta_norm is not None and self.max_delta_norm > 0:
            norm = delta.norm(p="fro")
            if norm > self.max_delta_norm:
                delta = delta * (self.max_delta_norm / (norm + 1e-6))
        return delta

    def reset_parameters(self):
        nn.init.zeros_(self.U)
        nn.init.normal_(self.V, std=0.1)


class StabilityGate(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, loss_val):
        dev = next(self.parameters()).device
        if not isinstance(loss_val, torch.Tensor):
            loss_val = torch.tensor(loss_val, device=dev, dtype=torch.float32)
        else:
            loss_val = loss_val.to(device=dev, dtype=torch.float32)

        if loss_val.dim() == 0:
            loss_val = loss_val.view(1, 1)
        elif loss_val.dim() == 1:
            loss_val = loss_val.view(-1, 1)
        elif loss_val.dim() != 2:
            raise ValueError(f"loss_val must be scalar/1D/2D, got {tuple(loss_val.shape)}")
        return self.gate(loss_val)
