import json
from jiwer import wer, mer, wil, wip, cer
from tabulate import tabulate

# Function to calculate metrics
def calculate_metrics(groundtruth, transcription):
    return {
        'WER': wer(groundtruth, transcription),
        'MER': mer(groundtruth, transcription),
        'WIL': wil(groundtruth, transcription),
        'WIP': wip(groundtruth, transcription),
        'CER': cer(groundtruth, transcription)
    }

# Define the path to your JSON file
json_file_path = r'transcription_data.json'

# Load the transcription data from a JSON file
with open(json_file_path, 'r') as f:
    transcription_data = json.load(f)

# Initialize the table data with headers
table_data = [['Sample', 'Model', 'WER', 'MER', 'WIL', 'WIP', 'CER']]

# Process each entry and calculate metrics for each service's transcription
for entry in transcription_data:
    sample_name = entry['sample']
    # Add a row for each model's metrics
    assemblyai_metrics = calculate_metrics(entry['groundtruth'], entry['assemblyai_transcription'])
    whisperai_metrics = calculate_metrics(entry['groundtruth'], entry['whisperAI_transcription'])
    google_cloud_metrics = calculate_metrics(entry['groundtruth'], entry['google_cloud_stt_transcription'])
    aws_transcriber_metrics = calculate_metrics(entry['groundtruth'], entry['aws_transcriber_transcription'])
    wac2vec2_metrics = calculate_metrics(entry['groundtruth'], entry['wav2vec2_large_xlsr_53_dutch_transcription'])
    azure_metrics = calculate_metrics(entry['groundtruth'], entry['azure_transcription'])

    # Add formatted metrics for each model to the table data
    table_data.append([sample_name, 'assemblyAI'] + [round(metric, 4) for metric in assemblyai_metrics.values()])
    table_data.append(['', 'whisperAI'] + [round(metric, 4) for metric in whisperai_metrics.values()])
    table_data.append(['', 'google_cloud_stt'] + [round(metric, 4) for metric in google_cloud_metrics.values()])
    table_data.append(['', 'aws_transcriber'] + [round(metric, 4) for metric in aws_transcriber_metrics.values()])
    table_data.append(['', 'wav2vec2'] + [round(metric, 4) for metric in wac2vec2_metrics.values()])
    table_data.append(['', 'azure'] + [round(metric, 4) for metric in azure_metrics.values()])

# Print the table using tabulate
print(tabulate(table_data, headers='firstrow', tablefmt='grid'))
