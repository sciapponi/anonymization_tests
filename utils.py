import torch 
from torch import nn
import librosa 
import numpy as np

class F0Extractor(nn.Module):

    def __init__(self,
                 sr,
                 fmin=40,
                 fmax=8000):
        super().__init__()
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
                                frame_length=1283
                                )
                energy = np.array(yin[-1])
                outs.append(torch.Tensor(np.array(yin[:-1])))
            
            outs.append(torch.Tensor(np.expand_dims(energy, 0)))
            output.append(torch.cat(outs))
        
        return torch.stack(output)


if __name__=="__main__":
    s = torch.randn(1,1,1600)
    module = F0Extractor(sr=16000)
    print(module(s))