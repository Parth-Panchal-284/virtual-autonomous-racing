import torch.nn as nn
import torch

# ─────────────────────────────────────────────
# Network building blocks
# ─────────────────────────────────────────────

class CNNStream(nn.Module):
    """
    Deeper CNN for stacked grayscale frames from TrackMania.
    Wider filters (32->64->128->128) capture richer 3D perspective features
    compared to the original Nature-DQN (32->64->64).
    Input : (B, C, H, W)  pixel values in [0, 255]
    Output: (B, out_dim)
    """
    def __init__(self, in_channels: int, img_h: int, img_w: int, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,          64,  kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,          128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(128,         128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            flat  = self.conv(dummy).shape[1]
        self.proj = nn.Sequential(
            nn.Linear(flat, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x / 255.0))
 
 
class VecStream(nn.Module):
    """
    Larger MLP for vector observations (speed, gear, lidar history ...).
    With 8 history frames of lidar (~152 floats), the original 128->64
    was a severe bottleneck. Now 256->256->128 to preserve lidar detail.
    Input : (B, D)
    Output: (B, out_dim)
    """
    def __init__(self, vec_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vec_dim, 256), nn.Tanh(),
            nn.Linear(256,     256), nn.Tanh(),
            nn.Linear(256, out_dim), nn.Tanh(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DualStreamEncoder(nn.Module):
    """
    Fuses CNN visual features with MLP state features using a deeper fusion MLP.
    Two fusion layers instead of one give the network room to learn non-trivial
    interactions between visual context and vehicle state (e.g. speed + corner ahead).
    Either stream can be absent (pass None).
    Output: (B, enc_dim)
    """
    def __init__(
        self,
        cnn,
        vec,
        cnn_out_dim: int,
        vec_out_dim: int,
        enc_dim: int = 512,
    ):
        super().__init__()
        self.cnn = cnn
        self.vec = vec
        fusion_in = (cnn_out_dim if cnn is not None else 0) + \
                    (vec_out_dim if vec is not None else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, enc_dim), nn.ReLU(),
            nn.Linear(enc_dim,   enc_dim), nn.ReLU(),
        )
 
    def forward(self, img, vec):
        parts = []
        if self.cnn is not None and img is not None:
            parts.append(self.cnn(img))
        if self.vec is not None and vec is not None:
            parts.append(self.vec(vec))
        return self.fusion(torch.cat(parts, dim=-1))
