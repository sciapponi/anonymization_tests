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
        # print(x.shape)
        # print(f0.shape)
        # f0 = f0[...,:-1]
        # print(x.shape)
        # print(f0.shape)
        if len(x.shape)==2:
            x=x.unsqueeze(0)
        x = torch.cat((x, f0.squeeze(-2)), dim=1)
        x = self.first_conv(x)
        print("FIRST CONV" ,x.shape)
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

class LearnablePoolingParam(nn.Module):
    #Github implementation https://github.com/hrnoh24/stream-vc/blob/main/src/models/components/streamvc.py#L113
    def __init__(self, embedding_dim):
        
        super().__init__()
        self.learnable_query = nn.Parameter(torch.randn((1, 1, embedding_dim)))

    def forward(self, emb):
        
        B, d_k, _ = emb.shape
        # print("lq", self.learnable_query.shape)
        query = self.learnable_query.expand(B, -1, -1) # [B, 1, C]
        key = emb # [B, C, N]
        value = emb.transpose(1, 2) # [B, N, C]

        score = torch.matmul(query, key) # [B, 1, N]
        score = score / (d_k ** 0.5)

        probs = F.softmax(score, dim=-1) # [B, 1, N]
        out = torch.matmul(probs, value) # [B, 1, C]
        out = out.squeeze(1) # [B, C]

        return out
    
class ConvSpeakerEncoder(nn.Module):
    def __init__(self, input_dim, num_channels, output_dim):
        super(ConvSpeakerEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, num_channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1, padding=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(num_channels, num_channels)
        self.fc2 = nn.Linear(num_channels, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Transpose for 1D convolution
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Normalize the embedding
        normalized_embedding = F.normalize(x, p=2, dim=1)
        
        return normalized_embedding
    
class APCModule(nn.Module):

    def __init__(self, c_in, c_bank):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=1,
                    dilation=1,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=4,
                    dilation=4,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=6,
                    dilation=6,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=8,
                    dilation=8,
                    padding_mode="reflect",
                ),
            ])
        
        self.act = nn.ReLU()

    def forward(self, inp):
        out_list = []
        for layer in self.layers:
            print(self.act(layer(inp)).shape)
            out_list.append(self.act(layer(inp)))
        outData = torch.cat(out_list + [inp], dim=1)
        return outData

if __name__=="__main__":
    # Example usage
    input_dim = 40  # e.g., 40 mel-frequency cepstral coefficients (MFCCs)
    num_channels = 128
    output_dim = 64  # embedding dimension
    batch_size = 1
    sequence_length = 100

    model = ConvSpeakerEncoder(input_dim, num_channels, output_dim)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    output = model(dummy_input)
    print(output.shape)  # Should be (batch_size, output_dim)

    APC =APCModule(80, 100)
    print(APC(torch.randn(1, 80,100)).shape)