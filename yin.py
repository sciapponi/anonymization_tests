import torch 
from torch import nn
import torch.nn.functional as F
import torchaudio

class ParabolicInterpolation1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_a = torch.tensor([1, -2, 1], dtype=torch.float).view(1,1,3)
        self.kernel_b = torch.tensor([-0.5, 0, 0.5], dtype=torch.float).view(1,1,3)
    def forward(self, x):
        output = []
        for sample in x:
            sample = sample.squeeze().unsqueeze(-2)
            a = torch.nn.functional.conv1d(sample, weight=self.kernel_a, stride=1, padding=1)
            b = torch.nn.functional.conv1d(sample, weight=self.kernel_b, stride=1, padding=1)
            out = ((-b)/a).squeeze(-2)
            out[...,0] = 0
            out[...,-1] = 0
            mask = (torch.abs(b)>=torch.abs(a)).squeeze(-2)

            out = torch.where(mask, torch.zeros_like(out), out)
            
            output.append(out)
        
        return torch.stack(output, dim=0)



class YinModule(nn.Module):
    def __init__(self,
                 fmin:float,
                 fmax: float,
                 sr: float = 22050,
                 frame_length: int = 2048,
                 win_length=None,
                 hop_length = None,
                 threshold:float = 0.1,
                 center: bool = True,
                 pad_mode = "constant"):
        super().__init__()
        if fmin is None or fmax is None:
            raise ValueError('both "fmin" and "fmax" must be provided')
        
        self.fmin, self.fmax = fmin, fmax
        self. sr = sr
        self.threshold = threshold
        self.center = center 
        self.pad_mode = pad_mode
        self.frame_length = frame_length

        if win_length is None:
            self.win_length = frame_length // 2
        else:
            self.win_length = win_length

        if hop_length is None:
            self.hop_length = frame_length // 4
        else:
            self.hop_length = hop_length

        self.pi = ParabolicInterpolation1d()
    
   
    def frame_audio(self, audio_tensor, frame_length, hop_length):
        """
        Frames a 2D tensor of 1D audio tensors (single channel) into chunks with specified hop length.

        Args:
            audio_tensor: A 2D tensor where the first dimension represents audio in the batch, 
                          and the second dimension represents audio samples (single channel).
            frame_length: Length (in samples) of each frame.
            hop_length: Hop length (in samples) between successive frames.

        Returns:
            A 3D tensor where the first dimension represents audio in the batch, 
            the second dimension remains 1 (representing a single channel), 
            and the following dimensions are the framed audio chunks.
        """

        if audio_tensor.dim() != 2:
            raise ValueError("Input audio must be a 2D tensor with shape [batch, audio_length].")

        audio_length = audio_tensor.shape[1]  # Audio samples in dimension 1

        if audio_length < frame_length:
            raise ValueError("Audio signals are shorter than frame length.")

        if hop_length <= 0:
            raise ValueError("Hop length must be positive.")

        # Calculate number of frames per audio
        num_frames_per_audio = 1 + (audio_length - frame_length) // hop_length

        # Unfold the audio with specified frame_length and hop_length
        framed_audio = audio_tensor.unfold(dimension=1, size=frame_length, step=hop_length)

        return framed_audio
    
    def find_local_minima(self, x):
        """
        Finds indices of local minima in a 1D PyTorch tensor using sign change detection.

        Args:
            x: A 1D PyTorch tensor.

        Returns:
            A tensor containing indices of local minima.
        """
        # Calculate difference between consecutive elements
        differences = x[...,1:] - x[..., :-1]
        # Detect sign changes (from negative to positive)
        sign_changes = (differences[...,:-1] < 0) & (differences[..., 1:] >= 0)
        sign_changes = F.pad(sign_changes, (1,1), value=False)

        return sign_changes
    

    def cumulative_mean_normalized_difference(self, 
                                              y_frames: torch.Tensor,
                                              frame_length: int,
                                              win_length:int,
                                              min_period:int,
                                              max_period:int,
                                              epsilon = 1.1754943508222871e-38
                                              ) -> torch.Tensor:
        # Autocorrelation
        y_frames = y_frames#.unsqueeze(0) # remove after making batched framing
        a = torch.fft.rfft(y_frames, frame_length, dim=-1)
        b = torch.fft.rfft(y_frames[..., :, 1:win_length+1].flip(2), frame_length, dim=-1)
        acf_frames = torch.fft.irfft(a*b, frame_length, dim=-1)[..., :, win_length:]
        acf_frames[torch.abs(acf_frames) < 1e-6] = 0

        # Energy terms
        energy_frames = torch.cumsum(y_frames**2, axis = -1)
        energy_frames = energy_frames[..., win_length:] - energy_frames[..., :-win_length]
        energy_frames[torch.abs(energy_frames)<1e-6] = 0
        energy_frames_0 = energy_frames[...,0]

        # Difference Function
        yin_frames = energy_frames[..., :1] + energy_frames - 2 *acf_frames

        # Cumulative mean normalized difference
        yin_numerator = yin_frames[..., min_period : max_period+1]

        # Broadcast to have leading ones
        tau_range = torch.arange(1, max_period+1, 1)[None, None, :]
        cumulative_mean = torch.cumsum(yin_frames[..., 1:max_period+1], dim=-1)/tau_range
        yin_denominator = cumulative_mean[..., min_period-1:max_period]
        yin_frames = yin_numerator / (yin_denominator + epsilon)
        return yin_frames, energy_frames_0

    def forward(self, y):
       
        if self.center:
            padding = [0, 0] * y.ndim
            padding[-2], padding[-1] = (self.frame_length // 2, self.frame_length // 2)
            padding.reverse() # torch padding dims are reversed w.r.t. numpy
            padding = tuple(padding)
            y = torch.nn.functional.pad(y, padding, mode=self.pad_mode)
        
        #frame_audio
        y_frames = self.frame_audio(y.squeeze(), self.frame_length, self.hop_length)#.permute(-1,-2)

        # Calculate minimum and maximum periods
        min_period = int(torch.floor(torch.Tensor([self.sr / self.fmax])))

        max_period = int(torch.min(torch.ceil(torch.tensor([self.sr / self.fmin])), torch.Tensor([self.frame_length - self.win_length - 1])))
        
        yin_frames, energy_frames_0 = self.cumulative_mean_normalized_difference(y_frames, 
                                                                                 self.frame_length, 
                                                                                 self.win_length, 
                                                                                 min_period, 
                                                                                 max_period)

        parabolic_shifts = self.pi(yin_frames)

        is_trough = self.find_local_minima(yin_frames)
        is_trough[..., 0] = yin_frames[..., 0] < yin_frames[...,1]

        # Minima peak below threshold
        is_threshold_through = torch.logical_and(is_trough, yin_frames < self.threshold)

        # Absolute threshold

        target_shape = list(yin_frames.shape)
        target_shape[-1] = 1

        global_min = torch.argmin(yin_frames, dim=-1) # TO CHECK, global minima are not the same as Yin librosa (really similar though)
        yin_period = torch.argmax(is_threshold_through.int(), axis=-1)
        # print(yin_period)
        global_min = global_min.reshape(target_shape)
        yin_period = yin_period.reshape(target_shape)

        no_trough_below_threshold = torch.all(torch.logical_not(is_threshold_through), axis=-1)
        aperiodic = no_trough_below_threshold.int()
        yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

        yin_period = (
        min_period
        + yin_period
        + torch.take_along_dim(parabolic_shifts, yin_period, axis=-1)
        )[..., 0]

        f0 = self.sr/yin_period

        return f0, yin_period, aperiodic, energy_frames_0


if __name__ == "__main__":
    signal, fs = torchaudio.load("LJ001-0013.wav")
    new_fs = 16000
    signal = torchaudio.transforms.Resample(fs, new_fs)(signal)
    fs = new_fs
    signal = signal[:,:1600]
    signal = torch.stack([signal, signal, signal], dim=0)
    print(signal.shape)

    yin = YinModule(fmin =20, fmax=8000, sr=16000)
    print("\n---------------------------------\n")
    print(yin(signal))

