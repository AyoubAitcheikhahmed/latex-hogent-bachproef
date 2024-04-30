
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


speech, rate = librosa.load("sample1.wav",sr=1600)

display.Audio("sample1.wav",autoplay=true)
input_values = tokenizer(speech,return_tensors = 'pt').input_values

predicted_ids[0] = torch.argmax(logits, dim=1)

transcription = tokenizer.decode(predicted_ids)
print(transcription)