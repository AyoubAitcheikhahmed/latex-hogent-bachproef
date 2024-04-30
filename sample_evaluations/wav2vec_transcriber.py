import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-dutch")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-dutch")

#audio_path = "audio_samples/sample1_antwerpen.wav"
#audio_path = "audio_samples/sample3_brugge.wav"
#audio_path = "audio_samples/sample2_limburg.wav"
#audio_path = "audio_samples/sample5_algemeenNederlands.wav"
audio_path = "audio_samples/sample4_gent.wav"

waveform, sample_rate = torchaudio.load(audio_path)

if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription[0])