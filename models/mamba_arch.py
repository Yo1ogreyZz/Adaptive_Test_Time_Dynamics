import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


def selective_scan_ref(x, dt, A, B, C, D, z=None, delta_softplus=False):
    """
    Pure PyTorch implementation of selective scan (SSM recurrence).
    Matches the interface of mamba_ssm.selective_scan_fn.

    Args:
        x:  (B, D, L)   input
        dt: (B, D, L)   delta (time step)
        A:  (D, N)       state transition
        B:  (B, N, L)    input matrix
        C:  (B, N, L)    output matrix
        D:  (D,)         skip connection
        z:  (B, D, L)    gate (optional)
        delta_softplus: apply softplus to dt
    Returns:
        y:  (B, D, L)
    """
    B_batch, D_dim, L = x.shape
    N = A.shape[1]

    if delta_softplus:
        dt = F.softplus(dt)

    # Discretize: dA = exp(dt * A), dB_x = dt * x * B
    # dt: (B, D, L) -> (B, D, L, 1)
    # A:  (D, N)    -> (1, D, 1, N)
    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))  # (B, D, L, N)

    # x * dt: (B, D, L) -> (B, D, L, 1)
    # B: (B, N, L) -> (B, 1, L, N)
    dB_x = (dt * x).unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1)  # (B, D, L, N)

    # Sequential scan over time
    h = torch.zeros(B_batch, D_dim, N, device=x.device, dtype=x.dtype)  # (B, D, N)
    ys = []
    for t in range(L):
        h = dA[:, :, t, :] * h + dB_x[:, :, t, :]            # (B, D, N)
        y_t = (h * C[:, :, t].unsqueeze(1)).sum(-1)           # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=2)  # (B, D, L)

    # Skip connection
    if D is not None:
        y = y + x * D.unsqueeze(0).unsqueeze(-1)

    # Output gate
    if z is not None:
        y = y * F.silu(z)

    return y


class ATTD_MambaBlock(nn.Module):
    """
    Single Mamba block with ATTD delta_A injection support.
    Dimension flow: d_model -> d_inner -> d_model (via out_proj).
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.drop = nn.Dropout(dropout)

        # Input projection (d_model -> 2 * d_inner for x and z branches)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **factory_kwargs)

        # Causal depthwise convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, **factory_kwargs,
        )

        # SSM parameter projections: dt_rank=1 (simplified), B, C
        self.x_proj = nn.Linear(self.d_inner, 1 + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True, **factory_kwargs)

        # A matrix (log-parameterized so that exp(A_log) > 0, then negated)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        # Output projection (d_inner -> d_model) to allow layer stacking
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, hidden_states, delta_A=None):
        """
        Args:
            hidden_states: (B, L, d_model)
            delta_A: optional (d_inner, d_state) low-rank perturbation to A
        Returns:
            (B, L, d_model)
        """
        batch, seqlen, dim = hidden_states.shape

        # Input projection and split into x (SSM branch) and z (gate branch)
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        # Causal conv + SiLU activation
        x = self.conv1d(x)[:, :, :seqlen]
        x = F.silu(x)

        # Compute SSM parameters (dt, B, C) from x
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) n -> b n l", l=seqlen)
        C = rearrange(C, "(b l) n -> b n l", l=seqlen)

        # State transition matrix A, with optional ATTD injection
        A = -torch.exp(self.A_log.float())
        if delta_A is not None:
            A = A + delta_A

        # Selective scan
        if selective_scan_fn is not None:
            y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=z, delta_softplus=True)
        else:
            y = selective_scan_ref(x, dt, A, B, C, self.D.float(), z=z, delta_softplus=True)
        y = rearrange(y, "b d l -> b l d")

        return self.drop(self.out_proj(y))


class MambaBackbone(nn.Module):
    """
    Pure feature extractor: N x (MambaBlock + residual + LayerNorm).
    No embedding, no classification head -- those are task-specific.

    Input:  (B, L, d_model)
    Output: (B, L, d_model)
    """

    def __init__(self, n_layers, d_model, d_state=16, d_conv=4, expand=2,
                 dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList([
            ATTD_MambaBlock(d_model, d_state=d_state, d_conv=d_conv,
                            expand=expand, dropout=dropout, **factory_kwargs)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model, **factory_kwargs)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, x, delta_As=None):
        """
        Args:
            x: (B, L, d_model)  -- pre-embedded features
            delta_As: optional list of per-layer delta_A tensors, length n_layers
        Returns:
            (B, L, d_model)
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            dA = delta_As[i] if delta_As is not None else None
            x = x + layer(norm(x), delta_A=dA)
        return self.norm_f(x)