import torch 
from torch import nn
from vector_quantize_pytorch import ResidualVQ
from phi import Encoder as PhiEncoder 
from soundstream.decoder import Decoder as SoundStreamDecoder
from typing import Literal 
from torchsummary import summary 

class SoundPhi(nn.Module):

    def __init__(self, 
                 latent_space_dim,
                 n_q,
                 codebook_size):
        
        super().__init__()
        self.encoder = PhiEncoder(C=32,D=latent_space_dim)
        self.decoder = SoundStreamDecoder(C=40, D=latent_space_dim)

        self.quantizer = ResidualVQ(
            num_quantizers=n_q,
            codebook_size=codebook_size,
            dim=latent_space_dim,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2
        )

    def forward(
            self,
            x,
            mode: Literal['end-to-end', 'encode', 'decode'] = 'end-to-end',
        ):
        # x: batch_size x 1 x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x 1 x (T / 1)

        if mode == 'end-to-end':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            o = self.decoder(quantized.permute((0,2,1)))
            return o
        
        if mode == 'encode':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            return quantized
        
        if mode == 'decode':
            o = self.decoder(x.permute((0,2,1)))
            return o
        
if __name__=="__main__":
    net = SoundPhi(latent_space_dim=64,
                    n_q=16,
                    codebook_size=1024).cuda()
    
    # summary(net, (1,16000))
    print(net(torch.randn(1,1,16000).cuda()).shape)
    