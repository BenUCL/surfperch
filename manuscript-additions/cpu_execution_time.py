
import numpy as np
import os
import time
import tensorflow as tf
from chirp.inference import embed_lib
from chirp.inference import interface
from chirp.inference import models
from ml_collections import config_dict

# By default TF gobbles the max num of threads, meaning VGGish and YAMNet will use this. So standardise threads here.
num_threads = 1
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

# Find base diectory path
base_dir = os.getenv('BASE_DIR')
if not base_dir:
    raise ValueError("BASE_DIR environment variable is not set.")

# BirdNET model path
birdnet_model_path = os.path.join(base_dir, 'ucl_perch/models/birdnet/V2.3/BirdNET_GLOBAL_3K_V2.3_Model_FP32.tflite')
perch_model_path = os.path.join(base_dir, 'ucl_perch/models/perch')

# Create 10 min of audio
sample_rate = 16000
duration = 600
audio_sample = np.random.rand(duration * sample_rate).astype(np.float32)  # Generating random noise as a placeholder.

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
config.num_tflite_threads = num_threads
config.class_list_name = 'birdnet_v2_1'  
birdnet_model = models.BirdNet.from_config(config)

# Perch set up
config = config_dict.ConfigDict()
config.model_path = perch_model_path
config.sample_rate = 32000 
config.window_size_s = 5.0  
config.hop_size_s = 5.0   
perch_model = models.TaxonomyModelTF.from_config(config)

def batch_measure_inference_time_general(model: interface.EmbeddingModel, audio: np.ndarray, batch_size: int, window_size_s: float) -> float:
    segment_length = int(window_size_s * model.sample_rate)
    
    # Calculate the total length that is divisible by segment_length
    divisible_length = len(audio) - (len(audio) % segment_length)
    
    # Trim the audio to the nearest lower multiple of segment_length
    audio_trimmed = audio[:divisible_length]
    
    # Ensure the audio is segmented based on the window_size_s for the specific model
    audio_batch = audio_trimmed.reshape((-1, segment_length))
    
    # Process each batch
    total_time = 0.0
    for start_idx in range(0, audio_batch.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, audio_batch.shape[0])
        batch = audio_batch[start_idx:end_idx]
        
        # Padding the last batch if it is smaller than the batch size
        if batch.shape[0] < batch_size:
            padding = np.zeros((batch_size - batch.shape[0], segment_length))
            batch = np.concatenate([batch, padding], axis=0)
        
        start_time = time.time()
        model.batch_embed(batch)
        total_time += time.time() - start_time
    
    return total_time

# For a model expecting 5-second windows
yamnet_batch_time = batch_measure_inference_time_general(yamnet_model, audio_sample, 64, 1)
vggish_batch_time = batch_measure_inference_time_general(vggish_model, audio_sample, 64, 1)
birdnet_batch_time = batch_measure_inference_time_general(birdnet_model, audio_sample, 64, 3.0)
perch_batch_time = batch_measure_inference_time_general(perch_model, audio_sample, 64, 5.0)
print(f"YAMNet batch processing time: {yamnet_batch_time} seconds")
print(f"VGGish batch processing time: {vggish_batch_time} seconds")
print(f"BirdNET batch processing time: {birdnet_batch_time} seconds")
print(f"Perch batch processing time: {perch_batch_time} seconds")
