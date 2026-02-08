import torch
import torch.nn as nn


class MonotonicController(nn.Module):
    """
    Monotonic multi-threshold controller.
    Maps a scalar ponder signal (e.g. self-supervised loss) to an integer
    adaptation depth K_dyn in [0, k_max].
    Uses Straight-Through Estimator (STE) for gradient flow.
    """

    def __init__(self, k_max=5):
        super().__init__()
        self.k_max = k_max
        # Learnable thresholds, initialized as a linearly spaced sequence
        self.threshold_raw = nn.Parameter(torch.linspace(-1.0, 1.0, k_max))

    def get_thresholds(self):
        """Return monotonically increasing thresholds via sorting."""
        return torch.sort(self.threshold_raw)[0]

    def forward(self, ponder_signal):
        """
        Args:
            ponder_signal: scalar or (B,) tensor representing input difficulty
        Returns:
            k_dyn: (B,) float tensor in [0, k_max] (round for discrete steps)
        """
        thresholds = self.get_thresholds()

        if ponder_signal.dim() == 0:
            ponder_signal = ponder_signal.unsqueeze(0)

        # (B, 1) vs (k_max,) -> broadcast comparison
        s = ponder_signal.unsqueeze(-1)

        # Hard indicator (forward) with soft gradient (backward) via STE
        indicator = (s > thresholds).float()
        soft = torch.sigmoid(s - thresholds)
        ste_indicator = indicator + (soft - soft.detach())

        return torch.sum(ste_indicator, dim=-1)


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
