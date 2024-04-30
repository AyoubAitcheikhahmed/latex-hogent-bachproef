import whisper
import warnings
#audio_path = "sample1_antwerpen.wav"
#audio_path = "sample3_brugge.wav"
#audio_path = "audio_samples/sample2_limburg.wav"
#audio_path = "audio_samples/sample5_algemeenNederlands.wav"
audio_path = "audio_samples/sample4_gent.wav"
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

model = whisper.load_model("base")

result = model.transcribe(audio_path)

with open("whisper_transcription_sample4_gent.txt","w") as f:
    f.write(result["text"])
#DEPENDENCIES:
# pip install openai-whisper pip install warnings