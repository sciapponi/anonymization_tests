import torch 
import torch.nn as nn
import torch.nn.functional as F

class FilmLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.film = nn.Linear(in_features=in_features, out_features=2*out_features)

    def forward(self, x, conditioning):
        beta, gamma = self.film(conditioning).split(self.out_features, dim=-1)

        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)

class FilmedDecoder(nn.Module):
    # ADDS FILM LAYER TO SOUNDSTREAM DECODER

    def __init__(self, soundstream_decoder, C, conditioning_size):
        super().__init__()

        decoder_blocks = [j for j in soundstream_decoder.children()][0]
        self.first_conv = decoder_blocks[0]
        self.last_conv = decoder_blocks[-1]
        self.decoder_blocks = decoder_blocks[1:-1]

        # FILM layers

        self.films = nn.ModuleList([
                nn.ModuleList([FilmLayer(conditioning_size, i*C) for j in range(3)])
                for i in (8, 4, 2, 1)
        ])
    
    def forward(self, x, f0, conditioning):
        x = torch.cat((x, f0.cuda().squeeze(-2)), dim=1)
        x = self.first_conv(x)
        # PASS TROUGH FILM CONDITIONING LAYER BEFORE EACH RESIDUAL UNIT
        for i, sequential in enumerate(self.decoder_blocks):
            blocks = [j for j in sequential.children()][0]
            x = blocks[0](x) # First conv transposed
            for j, residual in enumerate(blocks[1:]):
                x = x.permute(0,-2,-1)
                x = self.films[i][j](x, conditioning)
                x = residual(x)

        x = self.last_conv(x)

        return x

class LearnablePooling(nn.Module):
    def __init__(self, embedding_dim):
        
        super().__init__()
        self.query = nn.Linear(embedding_dim, 1)

    def forward(self, speaker_frames):
        q = self.query(speaker_frames.permute(0,-1,-2))
        attention_scores = q.permute(0,-1,-2).softmax(1)
        context_vector = speaker_frames * attention_scores

        return torch.sum(context_vector, dim=-1)
