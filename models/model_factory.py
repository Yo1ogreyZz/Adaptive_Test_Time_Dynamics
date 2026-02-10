from .mamba_arch import MambaBackbone
from .attd_wrapper import ATTDWrapper


def build_attd_backbone(config):
    """
    Build a MambaBackbone wrapped with ATTDWrapper.

    Required config keys:
        n_layers, d_model
    Optional config keys:
        d_state (16), d_conv (4), expand (2),
        rank (8), k_max (5), inner_lr (1e-3)

    Returns:
        ATTDWrapper  -- input (B, L, D) -> output (B, L, D)
    """
    backbone = MambaBackbone(
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        d_state=config.get("d_state", 16),
        d_conv=config.get("d_conv", 4),
        expand=config.get("expand", 2),
        dropout=config.get("dropout", 0.0),
    )
    return ATTDWrapper(backbone, config)
