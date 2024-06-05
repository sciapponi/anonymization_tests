import torch 
from torch import nn
import librosa 
import numpy as np
from yin import YinModule

# class F0Extractor(nn.Module):

#     def __init__(self,
#                  sr,
#                  fmin=40,
#                  fmax=8000):
#         super().__init__()
#         self.sr = sr
#         self.fmin = fmin 
#         self.fmax = fmax

#     def forward(self, signal):
#         signal = signal.cpu().numpy()
#         output = []
#         for s in signal:
#             outs = []
#             for th in [0.05, 0.10, 0.15]:
#                 yin = librosa.yin(s, 
#                                 fmin=self.fmin, 
#                                 fmax =self.fmax, 
#                                 sr=self.sr,
#                                 trough_threshold=th,
#                                 frame_length=1283
#                                 )
#                 energy = np.array(yin[-1])
#                 outs.append(torch.Tensor(np.array(yin[:-1])))
            
#             outs.append(torch.Tensor(np.expand_dims(energy, 0)))
#             output.append(torch.cat(outs))
        
#         return torch.stack(output)
    
class F0Extractor(nn.Module):

    def __init__(self,
                 sr,
                 fmin=20,
                 fmax=8000,
                 frame_length=1283):
        super().__init__()

        # Yin Modules for threshilds: 0.15, 0.10, 0.05
        self.yin_module_005 = YinModule(fmin, fmax, sr, threshold=0.05, frame_length=frame_length)
        self.yin_module_010 = YinModule(fmin, fmax, sr, threshold=0.10, frame_length=frame_length)
        self.yin_module_015 = YinModule(fmin, fmax, sr, threshold=0.15, frame_length=frame_length)

    def whitening(self, x):
        return (x - x.mean())/x.std()
    
    def forward(self, signal):
        # Returns frames for each threshold
        f0_0, period_0, aperiodic_0, energy = self.yin_module_005(signal)
        f0_1, period_1, aperiodic_1, energy = self.yin_module_010(signal)
        f0_2, period_2, aperiodic_2, energy = self.yin_module_015(signal)

        out = torch.stack([self.whitening(f0_0), self.whitening(period_0), aperiodic_0, 
                           self.whitening(f0_1), self.whitening(period_1), aperiodic_1,
                           self.whitening(f0_2), self.whitening(period_2), aperiodic_2,
                           energy ], dim=1)
        return out


if __name__=="__main__":
    s = torch.randn(8,1,16000)
    module = F0Extractor(sr=16000)
    print(module(s))