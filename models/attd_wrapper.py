import torch
import torch.nn as nn
import torch.nn.functional as F
from .attd_adapter import LowRankAdapter, StabilityGate
from .controller import MonotonicController


class ATTDWrapper(nn.Module):
    """
    Mamba backbone + ATTD
    """

    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.n_layers = backbone.n_layers
        self.d_inner = backbone.d_inner
        self.d_state = backbone.d_state

        rank = config.get("rank", 8)
        max_delta_norm = config.get("max_delta_norm", 0.5)
        self.adapters = nn.ModuleList(
            [LowRankAdapter(self.d_inner, self.d_state, rank=rank, max_delta_norm=max_delta_norm) for _ in range(self.n_layers)]
        )

        self.controller = MonotonicController(
            k_max=config.get("k_max", 5),
            tau=config.get("tau", 0.1),
        )
        self.stability_gate = StabilityGate()

        self.inner_lr = config.get("inner_lr", 1e-3)
        self.lambda_cost = config.get("lambda_cost", 1e-3)
        self.lambda_ctrl = config.get("lambda_ctrl", 1.0)
        self.entropy_weight = config.get("entropy_weight", 1.0)
        self.kl_weight = config.get("kl_weight", 0.2)
        self.confidence_tau = config.get("confidence_tau", 0.0)
        self.adapter_grad_clip = config.get("adapter_grad_clip", 1.0)
        self.train_mode = config.get("train_mode", "base")

        self._task_head_ref = [None]
        self._task_head_mode = "sequence_cls"

    def set_task_head(self, task_head, mode="sequence_cls"):
        self._task_head_ref[0] = task_head
        self._task_head_mode = mode

    @property
    def task_head(self):
        return self._task_head_ref[0]

    def _get_delta_As(self):
        return [adapter() for adapter in self.adapters]

    def _reset_adapters(self):
        for adapter in self.adapters:
            adapter.reset_parameters()

    def _adapter_parameters(self):
        return [p for adapter in self.adapters for p in adapter.parameters()]

    def _classification_ttt_loss(self, logits, ref_logits=None):
        probs = F.softmax(logits, dim=-1)
        ent_per = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)

        if self.confidence_tau > 0:
            conf = probs.max(dim=-1).values
            mask = (conf >= self.confidence_tau).float()
            ent = (ent_per * mask).sum() / (mask.sum() + 1e-6)
        else:
            ent = ent_per.mean()

        if ref_logits is not None and self.kl_weight > 0:
            kl = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
        else:
            kl = 0.0

        return self.entropy_weight * ent + self.kl_weight * kl, ent_per

    def compute_internal_loss(self, x, delta_As=None, ref_logits=None, per_sample=False):
        out = self.backbone(x, delta_As=delta_As)

        if self.task_head is not None:
            if self._task_head_mode == "token_lm":
                token_logits = self.task_head(out)
                probs = F.softmax(token_logits, dim=-1)
                ent_per_token = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
                ent_per = ent_per_token.mean(dim=-1)
                if ref_logits is not None and self.kl_weight > 0:
                    kl = F.kl_div(
                        F.log_softmax(token_logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="batchmean",
                    )
                else:
                    kl = 0.0
                loss = self.entropy_weight * ent_per.mean() + self.kl_weight * kl
                if per_sample:
                    return ent_per
                return loss

            pooled = out.mean(dim=1)
            logits = self.task_head(pooled)
            loss, ent_per = self._classification_ttt_loss(logits, ref_logits=ref_logits)
            if per_sample:
                return ent_per
            return loss

        # fallback
        mse_per = F.mse_loss(out[:, :-1, :], x.detach()[:, 1:, :], reduction="none").mean(dim=(1, 2))
        if per_sample:
            return mse_per
        return mse_per.mean()

    def _decide_k_dyn(self, s):
        out = self.controller(s)
        if isinstance(out, (tuple, list)):
            K_soft = out[0].float()
        else:
            K_soft = out.float()

        # avoids dynamic-compute collapse and preserves input-wise difficulty allocation.
        k_each = torch.ceil(K_soft).long().clamp_(0, self.controller.k_max)
        return k_each, K_soft

    def inner_loop_adapt(self, x, k_dyn: int):
        if k_dyn <= 0:
            return None

        x_orig = x.detach()

        ref_logits = None
        if self.task_head is not None:
            with torch.no_grad():
                out_base = self.backbone(x_orig, delta_As=None)
                if self._task_head_mode == "token_lm":
                    ref_logits = self.task_head(out_base)
                else:
                    ref_logits = self.task_head(out_base.mean(dim=1))

        adapter_params = self._adapter_parameters()
        for p in adapter_params:
            p.requires_grad_(True)

        inner_optimizer = torch.optim.SGD(adapter_params, lr=self.inner_lr)

        with torch.enable_grad():
            for _ in range(k_dyn):
                inner_optimizer.zero_grad()
                delta_As = self._get_delta_As()
                loss = self.compute_internal_loss(
                    x_orig, delta_As=delta_As, ref_logits=ref_logits, per_sample=False
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(adapter_params, self.adapter_grad_clip)

                gamma = self.stability_gate(loss.detach()).mean().clamp(0.0, 1.0)
                with torch.no_grad():
                    for group in inner_optimizer.param_groups:
                        lr = group["lr"]
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            p_next = p - lr * p.grad
                            p.copy_((1.0 - gamma) * p + gamma * p_next)

        for p in adapter_params:
            p.requires_grad_(False)

        return [adapter().detach() for adapter in self.adapters]

    def _forward_samplewise_k(self, x, k_each):
       
        k_each = k_each.to(device=x.device)
        unique_k = torch.unique(k_each, sorted=True)
        out_full = None

        for k_val in unique_k.tolist():
            idx = torch.nonzero(k_each == k_val, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            x_sub = x.index_select(0, idx)

            if k_val <= 0:
                with torch.no_grad():
                    out_sub = self.backbone(x_sub, delta_As=None)
            else:
                self._reset_adapters()
                delta_As = self.inner_loop_adapt(x_sub, int(k_val))
                with torch.no_grad():
                    out_sub = self.backbone(x_sub, delta_As=delta_As)

            if out_full is None:
                out_full = torch.empty(
                    x.size(0), out_sub.size(1), out_sub.size(2), device=out_sub.device, dtype=out_sub.dtype
                )
            out_full.index_copy_(0, idx, out_sub)

        if out_full is None:
            out_full = self.backbone(x, delta_As=None)
        return out_full

    def forward(self, x, return_info: bool = False):
        if self.training and self.train_mode != "controller":
            out = self.backbone(x)
            if return_info:
                # return tensor k_dyn to unify downstream aggregation.
                return out, {
                    "k_dyn": torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                    "K_soft": torch.zeros(x.size(0), device=x.device),
                }
            return out

        if self.training and self.train_mode == "controller":
            with torch.no_grad():
                s_vec = self.compute_internal_loss(x, delta_As=None, per_sample=True)
            k_each, K_soft = self._decide_k_dyn(s_vec)
            out = self._forward_samplewise_k(x, k_each)

            info = {"s": float(s_vec.mean().item()), "k_dyn": k_each, "K_soft": K_soft}
            if return_info:
                return out, info
            return out

        # Eval: optional ATTD
        if getattr(self, "disable_attd", False):
            out = self.backbone(x)
            if return_info:
                return out, {
                    "s": 0.0,
                    "k_dyn": torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                    "K_soft": torch.zeros(x.size(0), device=x.device),
                }
            return out

        with torch.no_grad():
            s_vec = self.compute_internal_loss(x, delta_As=None, per_sample=True)

        if self.train_mode == "base":
            k_each = torch.full((x.size(0),), int(self.controller.k_max), dtype=torch.long, device=x.device)
            K_soft = k_each.float()
        else:
            k_each, K_soft = self._decide_k_dyn(s_vec)

        out = self._forward_samplewise_k(x, k_each)

        if out is None:
            out = self.backbone(x, delta_As=None)

        info = {"s": float(s_vec.mean().item()), "k_dyn": k_each, "K_soft": K_soft}
        if return_info:
            return out, info
        return out
