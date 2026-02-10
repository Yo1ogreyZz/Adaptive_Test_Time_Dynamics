import torch
import torch.nn as nn
import torch.nn.functional as F

class MonotonicController(nn.Module):
    """
    Monotonic multi-threshold controller.
    Maps a scalar ponder signal (e.g. self-supervised loss) to an integer
    adaptation depth K_dyn in [0, k_max].
    Uses Straight-Through Estimator (STE) for gradient flow.
    """

    def __init__(self, k_max=5, tau=0.1, init_scale=0.2):
        super().__init__()
        self.k_max = k_max
        
        self.tau = tau
        self.delta_raw = nn.Parameter(torch.full((k_max,), init_scale))

        self.t0 = nn.Parameter(torch.tensor(0.0))

    def get_thresholds(self):
        """Return monotonically increasing thresholds via sorting."""
        inc = F.softplus(self.delta_raw)
        t = self.t0 + torch.cumsum(inc, dim=0)
        return t

    def forward(self, ponder_signal):
        """
        Args:
            ponder_signal: scalar or (B,) tensor representing input difficulty
        Returns:
            k_dyn: (B,) float tensor in [0, k_max] (round for discrete steps)
        """
        thresholds = self.get_thresholds()

        dev = next(self.parameters()).device
        if not isinstance(ponder_signal, torch.Tensor):
            ponder_signal = torch.tensor(ponder_signal, device=dev, dtype=torch.float32)
        else:
            ponder_signal = ponder_signal.to(device=dev, dtype=torch.float32)

        if ponder_signal.dim() == 0:
            ponder_signal = ponder_signal.view(1)

        # (B, 1) vs (k_max,) -> broadcast comparison
        s = ponder_signal.unsqueeze(-1)

        # Hard indicator (forward) with soft gradient (backward) via STE
        soft = torch.sigmoid((s - thresholds) / self.tau)
        hard = (soft > 0.5).float()
        gates = hard.detach() - soft.detach() + soft

        K_soft = gates.sum(dim=-1)
        K_hard = hard.sum(dim=-1).long()

        return K_soft, K_hard, gates, thresholds


class PonderSignalProcessor(nn.Module):
    """
    Optional module to transform complex internal state (e.g. hidden state)
    into a scalar ponder signal. If input_dim is None, acts as identity.
    """

    def __init__(self, input_dim=None):
        super().__init__()
        if input_dim:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softplus(),
            )
        else:
            self.net = nn.Identity()

    def forward(self, x):
        return self.net(x).squeeze(-1)
