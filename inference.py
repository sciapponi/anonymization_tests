from experiment import Experiment
import torchaudio 
import torch 
# model =  Experiment()
path = "/home/ste/model_checkpoints/streamvc/new_attention_46.ckpt"

model = Experiment.load_from_checkpoint(path).cuda()

audio_1, sr =  torchaudio.load("/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0001.wav")
audio_2, sr =  torchaudio.load("/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0002.wav")
audio_3, sr =  torchaudio.load("/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0003.wav")
audio_4, sr =  torchaudio.load("/home/ste/Datasets/LJSpeech-1.1/wavs/LJ001-0004.wav")
target_audios = [ torchaudio.load(f"/home/ste/Datasets/LibriTTS/train-clean-100/60/121082/60_121082_000003_000000.wav")[0]for i in range(1,20)]
target_audio = torch.cat(target_audios, dim=1).unsqueeze(0)
# target_audio = torch.cat([audio_1, audio_2, audio_3, audio_4], dim=1).unsqueeze(0)
# target_audio = audio_1
target_audio = torchaudio.functional.resample(target_audio, sr, 16000)#[...,:96000]
target_audio *= 0.95 / torch.max(target_audio)
# target_audio =  torch.randn([1,1,48000])
print(target_audio.shape)

# test clean
# input_audio, sr = torchaudio.load("/home/ste/Datasets/LibriTTS/test-clean/1580/141084/1580_141084_000006_000000.wav")
input_audio, sr = torchaudio.load("/home/ste/Datasets/LibriTTS/test-clean/8224/274384/8224_274384_000001_000001.wav")
input_audio *= 0.95 / torch.max(input_audio)
# target_audio = torchaudio.functional.resample(target_audio, sr, 16000).unsqueeze(0)
# # target_audio = target_audio[...,:96000]
# print(target_audio.shape)
# train clean 100
# input_audio, sr = torchaudio.load("/home/ste/Datasets/LibriTTS/train-clean-100/4406/16883/4406_16883_000001_000003.wav")
input_audio = torchaudio.functional.resample(input_audio, sr, 16000)
input_audio = input_audio.unsqueeze(0)[...,:128000]

# target_audio = input_audio

print(input_audio.shape)
out = model.generate(input_audio.squeeze().unsqueeze(0).cuda(), input_audio.cuda())
out_target = model.generate(input_audio.squeeze().unsqueeze(0).cuda(), target_audio.cuda())
print((out-out_target).mean())
torchaudio.save("in.wav", input_audio.squeeze().unsqueeze(0).cpu(), 16000)
torchaudio.save("target.wav", target_audio.squeeze().unsqueeze(0).cpu(), 16000)
print(out)
print(out_target)
torchaudio.save("out.wav", out.squeeze().unsqueeze(0).cpu(), 16000)
torchaudio.save("out_target.wav", out_target.squeeze().unsqueeze(0).cpu(), 16000)