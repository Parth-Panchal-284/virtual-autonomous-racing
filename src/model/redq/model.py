import torch
import torch.nn as nn


class CNNStream(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
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
            dummy = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2])
            flat = self.conv(dummy).shape[1]

        self.proj = nn.Sequential(
            nn.Linear(flat, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x / 255.0)
        return self.proj(x)


class VecStream(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 256),
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







from model.effnet import EfficientNetStream


def build_encoder(
    img_shape: tuple[int, int, int],
    vec_dim: int,
    cnn_out: int = 512,
    vec_out: int = 128,
    enc_dim: int = 512,
    
    use_efficientnet: bool = False
    
) -> DualStreamEncoder:
    
    cnn = None
    if(use_efficientnet):
        try:
            cnn = EfficientNetStream(
                img_shape[1], img_shape[0],
                pretrained=True,
                freeze_features=True
            )
        except:
            print("Cannot use Effnet. Using original CNN Implmenetation.")
            cnn = CNNStream(img_shape, out_dim=cnn_out)
    else:
        cnn = CNNStream(img_shape, out_dim=cnn_out)

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
    return encoder