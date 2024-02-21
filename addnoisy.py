from predict_asr import AddNoise
from predict_asr import FunctionalModule
import torch
import torchaudio

audio_pipeline = torch.nn.Sequential(
    AddNoise(snr_target=7.5)
    if -7.5 is not None
    else FunctionalModule(lambda x: x),
    FunctionalModule(
        lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
    ),
)   

fn = '/ssd1/data/processed/FakeAVCeleb/audio/FakeVideo-FakeAudio/African/men/id00366/00118_id00076_Isiq7cA-DNE_faceswap_id01779_wavtolip.wav'
audio, _ = torchaudio.load(fn, normalize=True)
audio = audio.transpose(1, 0)
audio = audio_pipeline(audio)
audio = audio.transpose(0, 1)

filename = "output_audio.wav"

# 指定采样率
sample_rate = 16000  # 例如，44100Hz

# 保存音频文件
torchaudio.save(filename, audio, sample_rate)
