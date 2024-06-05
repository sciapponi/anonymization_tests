from experiment import Experiment
import torchaudio 
import torch 
# model =  Experiment()
path = "/home/ste/model checkpoints/streamvc143.ckpt"

model = Experiment.load_from_checkpoint(path).cuda()

audio_1, sr =  torchaudio.load("/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0001.wav")
audio_2, sr =  torchaudio.load("/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0002.wav")
audio_3, sr =  torchaudio.load("/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0003.wav")

target_audios = [ torchaudio.load(f"/home/ste/Datasets/LibriTTS/train-clean-100/60/121082/60_121082_000005_000007.wav")[0]for i in range(1,10)]
target_audio = torch.cat(target_audios, dim=1).unsqueeze(0)
# target_audio = torch.cat([audio_1, audio_2, audio_3], dim=1).unsqueeze(0)
target_audio = torchaudio.functional.resample(target_audio, sr, 16000)[...,:48000]
target_audio =  torch.randn([1,1,48000])*100
print(target_audio.shape)

input_audio, sr = torchaudio.load("/home/ste/Datasets/LibriTTS/test-clean/4970/29095/4970_29095_000003_000001.wav")
input_audio = torchaudio.functional.resample(input_audio, sr, 16000)
input_audio = input_audio.unsqueeze(0)[...,:48000]

print(input_audio.shape)
out = model.generate(input_audio.squeeze().unsqueeze(0).cuda(), target_audio.cuda())
print(out.shape)
torchaudio.save("out.wav", out.squeeze().unsqueeze(0).cpu(), 16000)