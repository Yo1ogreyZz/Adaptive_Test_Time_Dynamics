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

        self.n_layers = backbone.n_layers
        self.d_inner = backbone.d_inner
        self.d_state = backbone.d_state

        rank = config.get("rank", 8)

        # One adapter per Mamba layer
        self.adapters = nn.ModuleList([
            LowRankAdapter(self.d_inner, self.d_state, rank=rank)
            for _ in range(self.n_layers)
        ])

        self.controller = MonotonicController(
            k_max=config.get("k_max", 5),
            tau=config.get("tau", 0.1),
        )
        self.stability_gate = StabilityGate()
        self.inner_lr = config.get("inner_lr", 1e-3)

        self.lambda_cost = config.get("lambda_cost", 1e-3)

        self.train_mode = config.get("train_mode", "base")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_delta_As(self):
        """Compute delta_A from every layer's adapter."""
        delta_As = [adapter() for adapter in self.adapters]

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

    def _decide_k_dyn(self, s):
        """
        Decide integer K_dyn from ponder signal s.
        """
        out = self.controller(s)
        if isinstance(out, tuple) or isinstance(out, list):
            # New API assumed: K_soft, K_hard, ...
            K_soft = out[0]
            K_hard = out[1]
            # Use K_hard for execution
            k_dyn = int(K_hard.mean().item())
            return k_dyn, K_soft
        else:
            # Old API: single tensor
            k_dyn = int(torch.round(out).mean().item())
            return k_dyn, out

    def inner_loop_adapt(self, x, k_dyn:int):
        """
        Test-time inner loop: optimize adapter parameters on self-supervised loss.
        Returns a list of detached delta_A tensors (one per layer), or None.
        """
        if k_dyn <= 0:
            return None

        # Only optimize adapter parameters
        for adapter in self.adapters:
            adapter.train()
        inner_optimizer = torch.optim.SGD(self._adapter_parameters(), lr=self.inner_lr)
        
        for _ in range(k_dyn):
            inner_optimizer.zero_grad()
            delta_As = self._get_delta_As()
            loss = self.compute_internal_loss(x, delta_As)
            loss.backward()

            # Scale gradients by stability gate
            gamma = self.stability_gate(loss.detach())
            gamma_s = gamma.mean().clamp(0.0, 1.0)

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

        return [adapter().detach() for adapter in self.adapters]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, return_info:bool=FileExistsError):
        """
        Args:
            x: (B, L, D) pre-embedded input
        Returns:
            (B, L, D)
        """
        if self.training:
            if self.train_mode == "controller":
                for p in self.backbone.parameters():
                    p.requires_grad = False
                
                with torch.no_grad():
                    # psi^(0): adapters are reset (delta_A = 0) to compute s under base dynamics
                    self._reset_adapters()
                    s = self.compute_internal_loss(x, delta_As=None)
                
                k_dyn, K_soft = self._decide_k_dyn(s)

                loss = self.lambda_cost * (K_soft.mean() if hasattr(K_soft, "mean") else torch.tensor(float(k_dyn), device=x.device))

                info = {"s": float(s.detach().item()), "k_dyn": int(k_dyn)}
                if return_info:
                    return loss, info
                return loss
            
            out = self.backbone(x)
            if return_info:
                return out, {}
            return out

        # Eval mode: reset adapters, run inner loop, final inference
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
