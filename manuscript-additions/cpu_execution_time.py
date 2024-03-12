# Create audio, batch it then speed test models

import numpy as np
import os
import time
import tensorflow as tf
from chirp.inference import interface
from chirp.inference import models
from ml_collections import config_dict

# Standardise threads used here.
num_threads = 16
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

# Find base diectory path
base_dir = os.getenv('BASE_DIR')
if not base_dir:
    raise ValueError("BASE_DIR environment variable is not set.")

# BirdNET model path
birdnet_model_path = os.path.join(base_dir, 'ucl_perch/models/birdnet/V2.3/BirdNET_GLOBAL_3K_V2.3_Model_FP32.tflite')
perch_model_path = os.path.join(base_dir, 'ucl_perch/models/perch')

# Create audio sample
sample_rate = 16000
duration = 3600
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

def time_batch_inference(model: interface.EmbeddingModel, audio: np.ndarray, batch_size: int, window_size_s: float) -> float:
    """
    Measures the total inference time for processing audio batches using the specified model.

    Args:
    - model: The model implementing the batch_embed method for batch processing.
    - audio: 1D numpy array of audio samples, normalized between -1.0 and 1.0.
    - batch_size: Number of audio segments per batch.
    - window_size_s: Duration of each audio segment in seconds, defining segment length.

    The function trims the audio to be divisible by the segment length, reshapes it into batches,
    applies padding if necessary, and aggregates the processing time for all batches.

    Returns:
    - The total processing time in seconds for all batches.
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
    for start_idx in range(0, audio_batch.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, audio_batch.shape[0])
        batch = audio_batch[start_idx:end_idx]
        
        # Apply padding to the last batch if necessary
        if needs_padding and end_idx >= audio_batch.shape[0]:
            batch = np.concatenate([batch, padding], axis=0)
        
        start_time = time.time()
        model.batch_embed(batch)
        total_time += time.time() - start_time
    
    return total_time

# For a model expecting 5-second windows
yamnet_batch_time = time_batch_inference(yamnet_model, audio_sample, 32, 1)
vggish_batch_time = time_batch_inference(vggish_model, audio_sample, 32, 1)
birdnet_batch_time = time_batch_inference(birdnet_model, audio_sample, 32, 3.0)
perch_batch_time = time_batch_inference(perch_model, audio_sample, 32, 5.0)
print(f"YAMNet batch processing time: {yamnet_batch_time} seconds")
print(f"VGGish batch processing time: {vggish_batch_time} seconds")
print(f"BirdNET batch processing time: {birdnet_batch_time} seconds")
print(f"Perch batch processing time: {perch_batch_time} seconds")


"""
Conclusion:
- Used 1 hr of audio and 16 threads to be as close to real world as possible. 1 hr also provides a better
    average but found results did not respond consistantly if these were changed.
- Perch seems to have small set up cost which is negligible when inferencing over real world applicable 
    lengths (hrs not seconds to mins), but is the slowest overall.
- TODO: take an average

Results (run on: 13th Gen Intel(R) Core(TM) i9-13900H)
# 1min, 1 thread, batch size = 32
YAMNet batch processing time: 0.7852568626403809 seconds
VGGish batch processing time: 1.7051284313201904 seconds
BirdNET batch processing time: 0.8993425369262695 seconds
Perch batch processing time: 18.492371797561646 seconds

# 1min, 1 thread, batch size = 64
YAMNet batch processing time: 0.7237632274627686 seconds
VGGish batch processing time: 1.6222598552703857 seconds
BirdNET batch processing time: 1.7326409816741943 seconds
Perch batch processing time: 35.36127805709839 seconds

# 1min, 16 thread, batch size = 32
YAMNet batch processing time: 0.9655618667602539 seconds
VGGish batch processing time: 0.9875094890594482 seconds
BirdNET batch processing time: 1.0375113487243652 seconds
Perch batch processing time: 9.27872896194458 seconds

# 1min, 16 thread, batch size = 64
YAMNet batch processing time: 0.9360687732696533 seconds
VGGish batch processing time: 0.9795386791229248 seconds
BirdNET batch processing time: 2.038900375366211 seconds
Perch batch processing time: 13.970347881317139 seconds

# 10 min audio 1 thread, batch size = 32
YAMNet batch processing time: 4.531564235687256 seconds
VGGish batch processing time: 14.502497911453247 seconds
BirdNET batch processing time: 2.1539978981018066 seconds
Perch batch processing time: 35.88314342498779 seconds

# 10 min audio 1 thread, batch size = 64
YAMNet batch processing time: 4.718485116958618 seconds
VGGish batch processing time: 15.175119161605835 seconds
BirdNET batch processing time: 3.06388783454895 seconds
Perch batch processing time: 36.73869490623474 seconds

# 10 min 16 threads, batch size = 32
YAMNet batch processing time: 6.805267810821533 seconds
VGGish batch processing time: 9.90481686592102 seconds
BirdNET batch processing time: 2.9302189350128174 seconds
Perch batch processing time: 17.076513051986694 seconds

# 10 min 16 threads, batch size = 64
YAMNet batch processing time: 7.30564546585083 seconds
VGGish batch processing time: 9.924951791763306 seconds
BirdNET batch processing time: 4.025118112564087 seconds
Perch batch processing time: 15.414696455001831 seconds

# 1hr of audio 1 thread, batch size = 32
YAMNet batch processing time: 26.373422622680664 seconds
VGGish batch processing time: 84.35902094841003 seconds
BirdNET batch processing time: 8.598920345306396 seconds
Perch batch processing time: 198.33394289016724 seconds

# 1hr of audio 16 threads, batch size = 32 <--- USE THIS? 
YAMNet batch processing time: 44.121607065200806 seconds
VGGish batch processing time: 56.96380305290222 seconds
BirdNET batch processing time: 11.815330028533936 seconds
Perch batch processing time: 93.90980958938599 seconds

# 1hr of audio 16 threads, batch size = 64
YAMNet batch processing time: 44.54791569709778 seconds
VGGish batch processing time: 57.272849798202515 seconds
BirdNET batch processing time: 12.709004402160645 seconds
Perch batch processing time: 77.0139672756195 seconds
"""
