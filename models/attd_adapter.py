import torch
import torch.nn as nn


class LowRankAdapter(nn.Module):
    """
    Low-rank perturbation to the SSM state matrix: delta_A = U @ V.
    Produces a (d_inner, d_state) additive update injected into each Mamba block.
    """

    def __init__(self, d_inner, d_state, rank=8, device=None, dtype=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.rank = rank

        factory_kwargs = {"device": device, "dtype": dtype}
        # Small init so the initial perturbation is near zero
        self.U = nn.Parameter(torch.randn(d_inner, rank, **factory_kwargs) * 0.001)
        self.V = nn.Parameter(torch.randn(rank, d_state, **factory_kwargs) * 0.001)

    def forward(self):
        """Return delta_A of shape (d_inner, d_state)."""
        return torch.matmul(self.U, self.V)

    def reset_parameters(self):
        """Re-initialize for instance-specific adaptation (called per sample)."""
        nn.init.normal_(self.U, std=0.001)
        nn.init.normal_(self.V, std=0.001)


class StabilityGate(nn.Module):
    """
    Learnable gate that maps the current inner-loop loss to a scalar
    gamma in [0, 1], used to scale gradient updates for stability.
    """

    def __init__(self, input_dim=1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, loss_val):
        """
        Args:
            loss_val: scalar or (1,) tensor representing current inner loss
        Returns:
            gamma: scalar in [0, 1]
        """
        if not isinstance(loss_val, torch.Tensor):
            loss_val = torch.tensor([[loss_val]], dtype=torch.float32)
        if loss_val.dim() == 0:
            loss_val = loss_val.unsqueeze(0).unsqueeze(0)
        if loss_val.dim() == 1:
            loss_val = loss_val.unsqueeze(0)
        return self.gate(loss_val)
