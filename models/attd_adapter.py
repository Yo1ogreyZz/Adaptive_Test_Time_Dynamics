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
        # U starts at zero, V at small random so that:
        # - delta_A = U @ V = 0 (no initial perturbation)
        # - grad w.r.t. U = grad_delta_A @ V^T != 0 (gradient flows)
        self.U = nn.Parameter(torch.zeros(d_inner, rank, **factory_kwargs))
        self.V = nn.Parameter(torch.randn(rank, d_state, **factory_kwargs) * 0.1)

    def forward(self):
        """Return delta_A of shape (d_inner, d_state).""" 
        return torch.matmul(self.U, self.V)

    def reset_parameters(self):
        """Re-initialize for instance-specific adaptation (called per sample)."""
        nn.init.zeros_(self.U)
        # We only reset U to ensure the priors.
        # nn.init.normal_(self.V, std=0.1)


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
        dev = next(self.parameters()).device

        if not isinstance(loss_val, torch.Tensor):
            loss_val = torch.tensor(loss_val, device=dev, dtype=torch.float32)
        else:
            loss_val = loss_val.to(device=dev, dtype=torch.float32)

        if loss_val.dim() == 0:
            loss_val = loss_val.view(1, 1)
        elif loss_val.dim() == 1:
            loss_val = loss_val.view(-1, 1)
        elif loss_val.dim() == 2:
            pass
        else: 
            raise ValueError(f"loss_val must be scalar/1D/2D tensor, got shape {tuple(loss_val.shape)}")

        return self.gate(loss_val)
