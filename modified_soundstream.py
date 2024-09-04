from soundstream.units import CausalConv1d, CausalConvTranspose1d
from torch import nn 
import torch 

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x
    

class ResidualUnitIN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            pad_mode='reflect'
        ):
        super(ResidualUnitIN, self).__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                pad_mode=pad_mode,
            ),
            nn.ELU(),
            nn.InstanceNorm1d(out_channels),
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                pad_mode=pad_mode,
            ),
            nn.ELU(),
            nn.InstanceNorm1d(out_channels),
        )

    def forward(self, x):
        return x + self.layers(x)

class ResidualUnitADAIN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dilation,
            pad_mode='reflect'
        ):
        super(ResidualUnitIN, self).__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                pad_mode=pad_mode,
            ),
            nn.ELU(),
            nn.InstanceNorm1d(out_channels),
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                pad_mode=pad_mode,
            ),
            nn.ELU(),
            nn.InstanceNorm1d(out_channels),
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlockIN(nn.Module):
    def __init__(self, out_channels, stride):
        super(EncoderBlockIN, self).__init__()

        self.layers = nn.Sequential(
            ResidualUnitIN(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=1
            ),
            ResidualUnitIN(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=3
            ),
            ResidualUnitIN(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=9
            ),
            CausalConv1d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=stride * 2,
                stride=stride
            ),
        )

    def forward(self, x):
        return self.layers(x)
    
class EncoderIN(nn.Module):
    def __init__(self, C, D, strides=(2, 4, 5, 8)):
        super(EncoderIN, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            EncoderBlockIN(out_channels=2*C, stride=strides[0]),
            EncoderBlockIN(out_channels=4*C, stride=strides[1]),
            EncoderBlockIN(out_channels=8*C, stride=strides[2]),
            EncoderBlockIN(out_channels=16*C, stride=strides[3]),
            CausalConv1d(in_channels=16*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)