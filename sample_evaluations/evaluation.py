import json
import csv
from jiwer import wer, mer, wil, wip, cer

# Function to calculate metrics
def calculate_metrics(groundtruth, transcription):
    return [
        wer(groundtruth, transcription),
        mer(groundtruth, transcription),
        wil(groundtruth, transcription),
        wip(groundtruth, transcription),
        cer(groundtruth, transcription)
    ]

# Define the path to your JSON file
json_file_path = r'transcription_data.json'

# Load the transcription data from a JSON file
with open(json_file_path, 'r') as f:
    transcription_data = json.load(f)

# Initialize the table data
table_data = [['Sample', 'Model', 'WER', 'MER', 'WIL', 'WIP', 'CER']]

# Process each entry and calculate metrics for each service's transcription
for entry in transcription_data:
    sample_name = entry['sample']
    # Add a row for each model's metrics
    assemblyai_metrics = calculate_metrics(entry['groundtruth'], entry['assemblyai_transcription'])
    whisperai_metrics = calculate_metrics(entry['groundtruth'], entry['whisperAI_transcription'])
    google_cloud_metrics = calculate_metrics(entry['groundtruth'], entry['google_cloud_stt_transcription'])
    aws_transcriber_metrics = calculate_metrics(entry['groundtruth'], entry['aws_transcriber_transcription'])
    wav2vec2_metrics = calculate_metrics(entry['groundtruth'], entry['wav2vec2_large_xlsr_53_dutch_transcription'])
    azure_metrics = calculate_metrics(entry['groundtruth'], entry['azure_transcription'])

    # Add formatted metrics for each model to the table data
    table_data.append([sample_name, 'assemblyAI'] + assemblyai_metrics)
    table_data.append(['', 'whisperAI'] + whisperai_metrics)
    table_data.append(['', 'google_cloud_stt'] + google_cloud_metrics)
    table_data.append(['', 'aws_transcriber'] + aws_transcriber_metrics)
    table_data.append(['', 'wav2vec2'] + wav2vec2_metrics)
    table_data.append(['', 'azure'] + azure_metrics)

# Write the table data to a CSV file
csv_file_path = 'transcription_metrics.csv'
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(table_data)

print(f"Data has been successfully written to {csv_file_path}")
