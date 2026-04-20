import torch
import torch.nn as nn


class CNNStream(nn.Module):
    """
    CNN stream for stacked grayscale frames.
    Input:  (B, C, H, W)
    Output: (B, out_dim)
    """

    def __init__(self, in_channels: int, img_h: int, img_w: int, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            flat = self.conv(dummy).shape[1]

        self.proj = nn.Sequential(
            nn.Linear(flat, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x / 255.0))


class VecStream(nn.Module):
    """
    MLP stream for scalar/vector state observations.
    Input:  (B, D)
    Output: (B, out_dim)
    """

    def __init__(self, vec_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vec_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualStreamEncoder(nn.Module):
    """
    Fuses visual and vector streams.
    Either stream can be disabled by passing None.
    """

    def __init__(
        self,
        cnn: nn.Module | None,
        vec: nn.Module | None,
        cnn_out_dim: int,
        vec_out_dim: int,
        enc_dim: int = 512,
    ):
        super().__init__()
        self.cnn = cnn
        self.vec = vec

        fusion_in = (cnn_out_dim if cnn is not None else 0) + (vec_out_dim if vec is not None else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
        )

    def forward(self, img: torch.Tensor | None, vec: torch.Tensor | None) -> torch.Tensor:
        parts = []
        if self.cnn is not None and img is not None:
            parts.append(self.cnn(img))
        if self.vec is not None and vec is not None:
            parts.append(self.vec(vec))
        return self.fusion(torch.cat(parts, dim=-1))


def build_encoder(
    img_channels: int | None,
    img_h: int | None,
    img_w: int | None,
    vec_dim: int | None,
    cnn_out: int = 512,
    vec_out: int = 128,
    enc_dim: int = 512,
) -> tuple[DualStreamEncoder, int]:
    cnn = None
    if img_channels is not None and img_h is not None and img_w is not None:
        cnn = CNNStream(img_channels, img_h, img_w, out_dim=cnn_out)

    vec = None
    if vec_dim is not None and vec_dim > 0:
        vec = VecStream(vec_dim, out_dim=vec_out)

    encoder = DualStreamEncoder(
        cnn=cnn,
        vec=vec,
        cnn_out_dim=cnn_out if cnn is not None else 0,
        vec_out_dim=vec_out if vec is not None else 0,
        enc_dim=enc_dim,
    )
    return encoder, enc_dim
