# Create audio, batch it then speed test models over a batch size and num threads grid.

import argparse
import csv
import os
import numpy as np
import tensorflow as tf
import time
from ml_collections import config_dict
from chirp.inference import models, interface

# Use arg parsing to set NUM_THREADS
parser = argparse.ArgumentParser(description='Run CPU execution time tests.')
parser.add_argument('--num_threads', type=int, default=16, help='The number of threads to run the tests with.')
args = parser.parse_args()
NUM_THREADS = args.num_threads
print(f'Running with NUM_THREADS={NUM_THREADS}')

# Set num threads for TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# Batch sizes to iterate over
BATCH_SIZES = [8, 32, 64, 128]

# Find base diectory path
base_dir = os.getenv('BASE_DIR')
if not base_dir:
    raise ValueError("BASE_DIR environment variable is not set.")

# BirdNET model path
birdnet_model_path = os.path.join(base_dir, 'ucl_perch/models/birdnet/V2.3/BirdNET_GLOBAL_3K_V2.3_Model_FP32.tflite')
perch_model_path = os.path.join(base_dir, 'ucl_perch/models/perch')

# Path for the CSV results
csv_file_path = os.path.join(base_dir, 'ucl_perch/reefperch/manuscript-additions/cpu_execution_time.csv')
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# Create audio sample
sample_rate = 32000 
duration = 3600 # 3600 = 1hr
audio_sample = np.random.rand(duration * sample_rate).astype(np.float32)


def time_batch_inference(model: interface.EmbeddingModel, audio: np.ndarray, batch_size: int, window_size_s: float) -> float:
    """
    Measures the total inference time for processing audio batches using the specified model, excluding the first batch.

    Args:
    - model: The model implementing the batch_embed method for batch processing.
    - audio: 1D numpy array of audio samples, normalized between -1.0 and 1.0.
    - batch_size: Number of audio segments per batch.
    - window_size_s: Duration of each audio segment in seconds, defining segment length.

    The function trims the audio to be divisible by the segment length, reshapes it into batches,
    applies padding if necessary, and aggregates the processing time for all batches, excluding the first batch.

    Returns:
    - The total processing time in seconds for all batches excluding the first.
    """
    segment_length = int(window_size_s * model.sample_rate)
    
    # Trim audio to be divisible by the segment length
    divisible_length = len(audio) - (len(audio) % segment_length)
    audio_trimmed = audio[:divisible_length]
    
    # Ensure the audio is segmented based on the window_size_s for the specific model
    audio_batch = audio_trimmed.reshape((-1, segment_length))
    
    # Calculate padding for final batch if needed
    final_batch_size = audio_batch.shape[0] % batch_size
    needs_padding = final_batch_size > 0
    if needs_padding:
        padding_amount = batch_size - final_batch_size
        padding = np.zeros((padding_amount, segment_length))
    
    # Process each batch
    total_time = 0.0
    is_first_iteration = True  # Flag to check if it's the first iteration
    for start_idx in range(0, audio_batch.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, audio_batch.shape[0])
        batch = audio_batch[start_idx:end_idx]
        
        # Apply padding to the last batch if necessary
        if needs_padding and end_idx >= audio_batch.shape[0]:
            batch = np.concatenate([batch, padding], axis=0)
        
        if is_first_iteration:
            print('Compiling model...')
            # Skip timing for the first iteration to allow compilation
            is_first_iteration = False
            model.batch_embed(batch)
            print('Test batch complete. Timing subsequent batches...')
        else:
            start_time = time.time()
            model.batch_embed(batch)
            total_time += time.time() - start_time
    
    return total_time

# Record for the fastest execution
fastest_executions = {
    'YAMNet': {'time': float('inf'), 'batch_size': None, 'num_threads': None},
    'VGGish': {'time': float('inf'), 'batch_size': None, 'num_threads': None},
    'BirdNET': {'time': float('inf'), 'batch_size': None, 'num_threads': None},
    'Perch': {'time': float('inf'), 'batch_size': None, 'num_threads': None},
}

# Open the CSV file and start appending new results to it
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Batch Size', 'Num Threads', 'Execution Time'])

    # Iterate over the grid search parameters
    for batch_size in BATCH_SIZES:
        # VGGish set up
        vggish_model = models.TFHubModel.vggish()

        # YAMNet set up
        yamnet_model = models.TFHubModel.yamnet()

        # BirdNET set up
        config = config_dict.ConfigDict()
        config.model_path = birdnet_model_path
        config.sample_rate = 48000  
        config.window_size_s = 3.0  
        config.hop_size_s = 3.0  
        config.num_tflite_threads = NUM_THREADS
        config.class_list_name = 'birdnet_v2_1'  
        birdnet_model = models.BirdNet.from_config(config)

        # Perch set up
        config = config_dict.ConfigDict()
        config.model_path = perch_model_path
        config.sample_rate = 32000 
        config.window_size_s = 5.0  
        config.hop_size_s = 5.0   
        perch_model = models.TaxonomyModelTF.from_config(config)

        # Set model window size mapping for the loop
        model_details = [
            ('YAMNet', yamnet_model, 1),
            ('VGGish', vggish_model, 1),
            ('BirdNET', birdnet_model, birdnet_model.window_size_s),
            ('Perch', perch_model, perch_model.window_size_s),
        ]


        # Run the test for each model with its respective window size
        for model_name, model, window_size in model_details:
            print(f'Current model: {model_name}')
            execution_time = time_batch_inference(model, audio_sample, batch_size, window_size)
            print(f"Model: {model_name}, Batch Size: {batch_size}, Num Threads: {NUM_THREADS}, Execution Time: {execution_time:.4f} seconds")

            # Write to CSV
            writer.writerow([model_name, batch_size, NUM_THREADS, execution_time])
            
            # Check if this is the fastest execution for the model
            if execution_time < fastest_executions[model_name]['time']:
                fastest_executions[model_name] = {
                    'time': execution_time,
                    'batch_size': batch_size,
                    'num_threads': NUM_THREADS
                }

# Print the fastest configuration for each model
for model_name, details in fastest_executions.items():
    print(f"{model_name} - Fastest Execution: {details['time']} seconds, Batch Size: {details['batch_size']}, Num Threads: {details['num_threads']}")