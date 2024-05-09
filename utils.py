import torch 
from torch import nn
import librosa 
import numpy as np

class F0Extractor(nn.Module):

    def __init__(self,
                 sr,
                 fmin=40,
                 fmax=8000):
        super(F0Extractor).__init__()
        self.sr = sr
        self.fmin = fmin 
        self.fmax = fmax

    def forward(self, signal):
        signal = signal.cpu().numpy()
        output = []
        for s in signal:
            outs = []
            for th in [0.05, 0.10, 0.15]:
                yin = librosa.yin(s, 
                                fmin=self.fmin, 
                                fmax =self.fmax, 
                                sr=self.sr,
                                trough_threshold=th,
                                win_length=int(self.sr*0.02)
                                )
                outs.append(torch.Tensor(np.array(yin)))
            output.append(torch.cat(outs))
        
        return torch.stack(output)


