import torch
import torch.nn as nn
import torch.nn.functional as F
from .attd_adapter import LowRankAdapter, StabilityGate
from .controller import MonotonicController

class ATTDWrapper(nn.Module):
    """
    Wraps a MambaBackbone with the ATTD mechanism.
    """
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.n_layers = backbone.n_layers
        self.d_inner = backbone.d_inner
        self.d_state = backbone.d_state

        rank = config.get("rank", 8)
        self.adapters = nn.ModuleList(
            [LowRankAdapter(self.d_inner, self.d_state, rank=rank) for _ in range(self.n_layers)]
        )
        self.controller = MonotonicController(
            k_max=config.get("k_max", 5),
            tau=config.get("tau", 0.1),
        )
        self.stability_gate = StabilityGate()
        self.inner_lr = config.get("inner_lr", 1e-3)
        self.lambda_cost = config.get("lambda_cost", 1e-3)
        self.train_mode = config.get("train_mode", "base")

    def _get_delta_As(self):
        return [adapter() for adapter in self.adapters]

    def _reset_adapters(self):
        for adapter in self.adapters:
            adapter.reset_parameters()

    def _adapter_parameters(self):
        return [p for adapter in self.adapters for p in adapter.parameters()]

    def compute_internal_loss(self, x, delta_As=None):
        """
        Enhanced Ponder Signal: MSE + Feature Variance.
        """
        output = self.backbone(x, delta_As=delta_As)
        mse_loss = F.mse_loss(output[:, :-1, :], x.detach()[:, 1:, :])
        feat_var = torch.var(output, dim=1).mean()
        return mse_loss + 0.1 * feat_var

    def _decide_k_dyn(self, s):
        out = self.controller(s)
        if isinstance(out, (tuple, list)):
            K_soft, K_hard = out[0], out[1]
            k_dyn = int(K_hard.float().mean().item())
            return k_dyn, K_soft
        else:
            k_dyn = int(torch.round(out).mean().item())
            return k_dyn, out

    def inner_loop_adapt(self, x, k_dyn: int):
        if k_dyn <= 0:
            return None
        adapter_params = self._adapter_parameters()
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
                gamma = self.stability_gate(loss.detach())
                gamma_s = gamma.mean().clamp(0.0, 1.0)
                with torch.no_grad():
                    for group in inner_optimizer.param_groups:
                        lr = group["lr"]
                        for p in group["params"]:
                            if p.grad is None: continue
                            p_next = p - lr * p.grad
                            p.copy_((1.0 - gamma_s) * p + gamma_s * p_next)
        
        for adapter in self.adapters:
            adapter.eval()
        for p, flag in zip(adapter_params, saved_grad_flags):
            p.requires_grad_(flag)
        return [adapter().detach() for adapter in self.adapters]

    def forward(self, x, return_info: bool = False):
        if self.training:
            if self.train_mode == "controller":
                for p in self.backbone.parameters():
                    p.requires_grad = False
                with torch.no_grad():
                    self._reset_adapters()
                    s = self.compute_internal_loss(x, delta_As=None)
                k_dyn, K_soft = self._decide_k_dyn(s)
                delta_As = self.inner_loop_adapt(x, k_dyn)
                out = self.backbone(x, delta_As=delta_As)
                return out, K_soft

            out = self.backbone(x)
            return out

        # Skip ATTD inner loop when disabled (zero-shot baseline)
        if getattr(self, 'disable_attd', False):
            out = self.backbone(x)
            if return_info:
                return out, {"s": 0.0, "k_dyn": 0}
            return out

        self._reset_adapters()
        with torch.no_grad():
            s = self.compute_internal_loss(x, delta_As=None)

            # In "base" mode the controller is untrained, so use fixed k_max steps.
            # In "controller" mode the controller has been trained, so use adaptive k.
            if self.train_mode == "base":
                k_dyn = self.controller.k_max
            else:
                k_dyn, _ = self._decide_k_dyn(s)

        delta_As = self.inner_loop_adapt(x, k_dyn)
        out = self.backbone(x, delta_As=delta_As)

        if return_info:
            return out, {"s": float(s.detach().item()), "k_dyn": int(k_dyn)}
        return out
