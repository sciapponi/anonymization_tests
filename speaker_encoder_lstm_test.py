import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(SpeakerEncoder, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = h_n[-1]
        
        # Pass through linear layer
        embedding = self.linear(last_hidden)
        
        # Normalize the embedding
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        
        return normalized_embedding

# Example usage
input_dim = 40  # e.g., 40 mel-frequency cepstral coefficients (MFCCs)
hidden_dim = 256
output_dim = 64  # embedding dimension
batch_size = 32
sequence_length = 100

model = SpeakerEncoder(input_dim, hidden_dim, output_dim)
dummy_input = torch.randn(batch_size, sequence_length, input_dim)
output = model(dummy_input)
print(output.shape)  # Should be (batch_size, output_dim)