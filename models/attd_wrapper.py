import torch
import torch.nn as nn
import torch.nn.functional as F
from .attd_adapter import LowRankAdapter, StabilityGate
from .controller import MonotonicController


class ATTDWrapper(nn.Module):
    """
    Wraps a MambaBackbone with the ATTD mechanism:
      - One LowRankAdapter per layer
      - A shared MonotonicController to decide adaptation depth
      - A shared StabilityGate to regulate gradient magnitude

    Manages the test-time inner-loop adaptation.
    Input:  (B, L, D) -> Output: (B, L, D)
    """

    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.config = config

        n_layers = backbone.n_layers
        d_inner = backbone.d_inner
        d_state = backbone.d_state
        rank = config.get("rank", 8)

        # One adapter per Mamba layer
        self.adapters = nn.ModuleList([
            LowRankAdapter(d_inner, d_state, rank=rank)
            for _ in range(n_layers)
        ])

        self.controller = MonotonicController(k_max=config.get("k_max", 5))
        self.stability_gate = StabilityGate()
        self.inner_lr = config.get("inner_lr", 1e-3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_delta_As(self):
        """Compute delta_A from every layer's adapter."""
        return [adapter() for adapter in self.adapters]

    def _reset_adapters(self):
        """Reset all adapter parameters for instance-specific adaptation."""
        for adapter in self.adapters:
            adapter.reset_parameters()

    def _adapter_parameters(self):
        """Collect all adapter parameters into a single list."""
        return [p for adapter in self.adapters for p in adapter.parameters()]

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def compute_internal_loss(self, x, delta_As):
        """
        Self-supervised next-token prediction loss (MSE reconstruction).
        Both input and output have shape (B, L, D) thanks to out_proj in each block.
        """
        output = self.backbone(x, delta_As=delta_As)
        target = x.detach()
        return F.mse_loss(output[:, :-1, :], target[:, 1:, :])

    def inner_loop_adapt(self, x):
        """
        Test-time inner loop: optimize adapter parameters on self-supervised loss.
        Returns a list of detached delta_A tensors (one per layer), or None.
        """
        # Determine adaptation depth from initial loss
        with torch.no_grad():
            initial_loss = self.compute_internal_loss(x, delta_As=None)
            k_dyn = int(torch.round(self.controller(initial_loss)).item())

        if k_dyn <= 0:
            return None

        # Only optimize adapter parameters
        for adapter in self.adapters:
            adapter.train()
        inner_optimizer = torch.optim.SGD(self._adapter_parameters(), lr=self.inner_lr)

        with torch.enable_grad():
            for _ in range(k_dyn):
                inner_optimizer.zero_grad()
                delta_As = self._get_delta_As()
                loss = self.compute_internal_loss(x, delta_As)
                loss.backward()

                # Scale gradients by stability gate
                gamma = self.stability_gate(loss.detach())
                for p in self._adapter_parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(gamma)

                inner_optimizer.step()

        for adapter in self.adapters:
            adapter.eval()

        return [adapter().detach() for adapter in self.adapters]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: (B, L, D) pre-embedded input
        Returns:
            (B, L, D)
        """
        if self.training:
            return self.backbone(x)

        # Eval mode: reset adapters, run inner loop, final inference
        self._reset_adapters()
        delta_As = self.inner_loop_adapt(x)
        return self.backbone(x, delta_As=delta_As)
