import torch
from torch import nn
import torch.nn.functional as F
# import nemo.collections.asr as nemo_asr

class XVectorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        for param in self.speaker_model.parameters():
            param.requires_grad = False
        self.speaker_model.freeze()
    def forward(self, input, target):

        similarities = []
        for i in range(input.shape[0]):
            signal1 = input[i]
            signal2 = target[i]
            length= torch.tensor([int(signal1.shape[1])])
            logits, emb1 = self.speaker_model(input_signal=signal1, input_signal_length=length.cuda())
            length= torch.tensor([int(signal2.shape[1])])
            logits, emb2 = self.speaker_model(input_signal=signal2, input_signal_length=length.cuda())
            similarities.append(F.cosine_similarity(emb1,emb2).mean())

        return torch.Tensor(similarities).mean()
    
class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, window=torch.hann_window(s).cuda(), win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, window=torch.hann_window(s).cuda(), win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss / (12 - 6)


class ReconstructionLoss2(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, sample_rate, eps=1e-5):
        super().__init__()
        import torchaudio
        self.layers = nn.ModuleList()
        self.alpha = []
        self.eps = eps
        for i in range(6, 12):
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(2 ** i),
                win_length=int(2 ** i),
                hop_length=int(2 ** i / 4),
                n_mels=64)
            self.layers.append(melspec)
            self.alpha.append((2 ** i / 2) ** 0.5)

    def forward(self, input, target):
        loss = 0
        for alpha, melspec in zip(self.alpha, self.layers):
            x = melspec(input)
            y = melspec(target)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss