import azure.cognitiveservices.speech as speechsdk
import time
import datetime
speech_key = '0169013f1e1a40f7bbcf8993f1315130'
service_region = 'eastus'

# Initialize the speech configuration
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = 'nl-NL'  # Set the language code here

#audio_file = "audio_samples/sample1_antwerpen.wav"
#audio_file = "audio_samples/sample3_brugge.wav"
#audio_file = "audio_samples/sample2_limburg.wav"
#audio_file = "audio_samples/sample5_algemeenNederlands.wav"
audio_file = "audio_samples/sample4_gent.wav"
# Set up the audio configuration
audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

# Create a speech recognizer object
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# Create an empty list to store the transcription results
transcriptions = []

# Define an event handler for continuous recognition
def continuous_recognition_handler(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcriptions.append(evt.result.text)

# Start continuous recognition
speech_recognizer.recognized.connect(continuous_recognition_handler)
speech_recognizer.start_continuous_recognition()

# Wait for the recognition to complete
timeout_seconds = 19  # timeout value (in seconds)
timeout_expiration = time.time() + timeout_seconds

while time.time() < timeout_expiration:
    time.sleep(1)  # Adjust the sleep duration as needed

# Stop continuous recognition
speech_recognizer.stop_continuous_recognition()

# Combine transcriptions into a single string
transcription = ' '.join(transcriptions)
current_datetime = datetime.datetime.now()
current_minute = current_datetime.second
# Write the transcription to a file
output_file = "AzureTranscription" + str(current_minute) +  ".txt"
with open(output_file, "w") as file:
    file.write(transcription)

print("Transcription saved to: " + output_file)
