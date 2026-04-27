import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights,
    EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    EfficientNet
)

from typing import Callable, Any
 
# ── EfficientNet variant registry ────────────────────────────────────────────
 
_EFFICIENTNET_VARIANTS : dict[str, tuple[Callable[..., EfficientNet], Any, int, int]] = {
    "b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280, 224),
    "b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, 1280, 240),
    "b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, 1408, 260),
    "b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, 1536, 300),
}
 
 
class EfficientNetStream(nn.Module):
    """
    EfficientNet-based visual encoder for RL observations.
 
    Strips the classification head and adds a linear projection so the output
    dimension matches the rest of the encoder pipeline.
 
    Supports grayscale stacked frames (in_channels != 3) by replacing the
    first conv layer, which discards ImageNet weights for that layer only.
 
    Args:
        in_channels:  Number of input channels (e.g. 1 for grayscale, 3 for RGB,
                      k for k stacked grayscale frames).
        out_dim:      Output feature dimension.  Defaults to 512.
        variant:      EfficientNet variant – one of "b0" … "b3".  Defaults to "b0".
        pretrained:   Load ImageNet weights for the feature extractor.
                      Weights for the first conv are re-initialised when
                      in_channels != 3 so the rest of the network still benefits.
        freeze_features: Freeze the EfficientNet feature extractor entirely and
                      only train the projection head.  Useful when fine-tuning
                      from a strong pre-trained init.
    """
 
    def __init__(
        self,
        image_size: int,
        in_channels: int,   
        out_dim: int = 512,
        variant: str = "b0",
        pretrained: bool = False,
        freeze_features: bool = False,
    ):
        super().__init__()
 
        if variant not in _EFFICIENTNET_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(_EFFICIENTNET_VARIANTS)}"
            )
 
        factory, weights, feature_dim, size = _EFFICIENTNET_VARIANTS[variant]
        
        if(image_size != size):
            raise ValueError(
                f"EffNet Variant \"{variant}\" input size is {size}. The current observation size is {image_size}."
            )
 
        # Load backbone (with or without ImageNet weights)
        backbone = factory(weights=weights if pretrained else None)
 
        # --- adapt first conv for arbitrary in_channels ---
        # Conv2d inside ConvNormActivation
        first_conv: nn.Conv2d = backbone.features[0][0]   # type: ignore
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size, # type: ignore
                stride=first_conv.stride, # type: ignore
                padding=first_conv.padding, # type: ignore
                bias=False,
            )
            # If pretrained, average the RGB weights across the channel dim so the
            # rest of the network starts in a reasonable state.
            if pretrained:
                with torch.no_grad():
                    avg = first_conv.weight.mean(dim=1, keepdim=True)  # (C_out,1,k,k)
                    new_conv.weight.copy_(avg.expand(-1, in_channels, -1, -1))
            backbone.features[0][0] = new_conv # type: ignore
 
        # Drop the classification head; keep only the feature extractor + pool
        self.features = backbone.features        # ConvNormActivation stack
        self.pool     = backbone.avgpool         # AdaptiveAvgPool2d → (B, C, 1, 1)
 
        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad_(False)
 
        # Projection to a fixed-size embedding
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
            nn.ReLU(),
        )
 
        self._out_dim = out_dim
 
    @property
    def out_dim(self) -> int:
        return self._out_dim
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — raw pixel values in [0, 255]
        x = x / 255.0
        x = self.features(x)          # (B, feature_dim, h', w')
        x = self.pool(x)              # (B, feature_dim, 1, 1)
        x = x.flatten(1)              # (B, feature_dim)
        return self.proj(x)           # (B, out_dim)
