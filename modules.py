import torch 
from torch import nn 

class FilmLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.film = nn.Linear(in_features=in_features, out_features=2*out_features)

    def forward(self, x, conditioning):
        beta, gamma = self.film(conditioning).split(self.out_features, dim=-1)

        return gamma * x + beta

class FilmedDDecoder(nn.Module):
    # ADDS FILM LAYER TO SOUNDSTREAM DECODER

    def __init__(self, soundstream_decoder, C, conditioning_size):
        super().__init__()

        decoder_children = [block for block in soundstream_decoder.children()]
        decoder_blocks = [i for j in decoder_children[0] for i in j.children()]
        self.first_conv = decoder_blocks[0]
        self.last_conv = decoder_blocks[-1]
        self.decoder_blocks = decoder_blocks[1:-2]

        # FILM layers

        self.films = nn.ModuleList([
                nn.ModuleList([FilmLayer(conditioning_size, i*C) for j in range(3)])
                for i in (8, 4, 2, 1)
        ])
    
    def forward(self, x, conditioning):

        x = self.first_conv(x)

        # PASS TROUGH FILM CONDITIONING LAYER BEFORE EACH RESIDUAL UNIT
        for i, sequential in enumerate(self.decoder_blocks):
            x = sequential[0](x)
            for j, residual in enumerate(sequential[1:]):
                x = self.films[i][j](x, conditioning)
                x = residual(x)

        x = self.last_conv(x)

        return x
