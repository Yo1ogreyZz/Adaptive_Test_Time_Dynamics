import torch
import torch.nn as nn
import torch.nn.functional as F

from .attd_adapter import LowRankAdapter, StabilityGate
from .controller import MonotonicController


class ATTDWrapper(nn.Module):
    """
    Wraps a MambaBackbone with the ATTD mechanism:
      - One LowRankAdapter per layer (what to adapt: dynamics-only, low-rank psi={U,V})
      - A shared MonotonicController to decide adaptation depth (how much: K_dyn)
      - A shared StabilityGate for stability control (gamma in [0,1])

    Test-time pipeline (proposal-aligned):
      1) s = L_int(x; theta_base, psi^(0))
      2) K_dyn = C(s)
      3) inner-loop update only psi for K_dyn steps (TTT)
      4) final inference with adapted dynamics
    """

    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.config = config

        self.n_layers = backbone.n_layers
        self.d_inner = backbone.d_inner
        self.d_state = backbone.d_state

        rank = config.get("rank", 8)

        # One adapter per Mamba layer
        self.adapters = nn.ModuleList(
            [LowRankAdapter(self.d_inner, self.d_state, rank=rank) for _ in range(self.n_layers)]
        )

        # Controller (supports tau if your controller implementation accepts it)
        self.controller = MonotonicController(
            k_max=config.get("k_max", 5),
            tau=config.get("tau", 0.1),
        )

        self.stability_gate = StabilityGate()
        self.inner_lr = config.get("inner_lr", 1e-3)

        # Cost coefficient for adaptive computation (used in controller training mode)
        self.lambda_cost = config.get("lambda_cost", 1e-3)

        # Training behavior switch: "base" or "controller"
        self.train_mode = config.get("train_mode", "base")

    def _get_delta_As(self):
        """Compute delta_A from every layer's adapter."""
        delta_As = [adapter() for adapter in self.adapters]

        # Safety checks to avoid silent mismatch with backbone injection
        assert len(delta_As) == self.n_layers, "delta_As length must equal number of layers"
        for dA in delta_As:
            assert dA.shape == (self.d_inner, self.d_state), (
                f"delta_A shape mismatch: expected {(self.d_inner, self.d_state)}, got {tuple(dA.shape)}"
            )
        return delta_As

    def _reset_adapters(self):
        """Reset all adapter parameters for instance-specific adaptation."""
        for adapter in self.adapters:
            adapter.reset_parameters()

    def _adapter_parameters(self):
        """Collect all adapter parameters into a single list."""
        return [p for adapter in self.adapters for p in adapter.parameters()]



    def compute_internal_loss(self, x, delta_As):
        """
        Self-supervised internal loss L_int.
        Current proxy: next-step embedding reconstruction (MSE).
        """
        output = self.backbone(x, delta_As=delta_As)
        target = x.detach()
        return F.mse_loss(output[:, :-1, :], target[:, 1:, :])


    def _decide_k_dyn(self, s):
        """
        Decide integer K_dyn from ponder signal s.
        Compatible with:
          - new controller API: returns (K_soft, K_hard, ...)
          - old controller API: returns a single tensor
        """
        out = self.controller(s)
        if isinstance(out, (tuple, list)):
            K_soft = out[0]
            K_hard = out[1]
            k_dyn = int(K_hard.float().mean().item())
            return k_dyn, K_soft
        else:
            k_dyn = int(torch.round(out).mean().item())
            return k_dyn, out


    def inner_loop_adapt(self, x, k_dyn: int):
        """
        Test-time inner loop: optimize adapter params on L_int for k_dyn steps.
        Uses damped update controlled by gamma for stability.
        Wrapped with torch.enable_grad() so it works inside @torch.no_grad() contexts.
        """
        if k_dyn <= 0:
            return None

        adapter_params = self._adapter_parameters()

        # Save and temporarily enable requires_grad for adapter params
        # (they may have been frozen externally, e.g. during controller training)
        saved_grad_flags = [p.requires_grad for p in adapter_params]
        for p in adapter_params:
            p.requires_grad_(True)

        for adapter in self.adapters:
            adapter.train()

        inner_optimizer = torch.optim.SGD(adapter_params, lr=self.inner_lr)

        with torch.enable_grad():
            for _ in range(k_dyn):
                inner_optimizer.zero_grad()
                delta_As = self._get_delta_As()
                loss = self.compute_internal_loss(x, delta_As)
                loss.backward()

                # Stability gate gamma in [0,1] (use mean as scalar damping)
                gamma = self.stability_gate(loss.detach())
                gamma_s = gamma.mean().clamp(0.0, 1.0)

                # Damped update:
                #   psi_next = psi - lr * grad
                #   psi <- (1-gamma)*psi + gamma*psi_next
                with torch.no_grad():
                    for group in inner_optimizer.param_groups:
                        lr = group["lr"]
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            p_next = p - lr * p.grad
                            p.copy_((1.0 - gamma_s) * p + gamma_s * p_next)

        for adapter in self.adapters:
            adapter.eval()

        # Restore original requires_grad flags
        for p, flag in zip(adapter_params, saved_grad_flags):
            p.requires_grad_(flag)

        return [adapter().detach() for adapter in self.adapters]

    def forward(self, x, return_info: bool = False):
        """
        Args:
            x: (B, L, D) pre-embedded input
        Returns:
            - training base mode: (B, L, D)
            - training controller mode: loss (and optionally info)
            - eval: (B, L, D) (and optionally info)
        """


        if self.training:
            if self.train_mode == "controller":
                # Freeze backbone in controller training mode
                for p in self.backbone.parameters():
                    p.requires_grad = False

                with torch.no_grad():
                    self._reset_adapters()
                    s = self.compute_internal_loss(x, delta_As=None)

                k_dyn, K_soft = self._decide_k_dyn(s)

                # Cost term: lambda * Cost(K). Prefer K_soft if differentiable.
                loss = self.lambda_cost * (K_soft.mean() if hasattr(K_soft, "mean") else torch.tensor(float(k_dyn), device=x.device))

                info = {"s": float(s.detach().item()), "k_dyn": int(k_dyn)}
                return (loss, info) if return_info else loss

            # Default: base training
            out = self.backbone(x)
            return (out, {}) if return_info else out


        self._reset_adapters()

        with torch.no_grad():
            s = self.compute_internal_loss(x, delta_As=None)
            k_dyn, _ = self._decide_k_dyn(s)

        delta_As = self.inner_loop_adapt(x, k_dyn)
        out = self.backbone(x, delta_As=delta_As)

        if return_info:
            info = {"s": float(s.detach().item()), "k_dyn": int(k_dyn)}
            return out, info
        return out
