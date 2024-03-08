import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time  # Import the time module

# # Creat 5min of audio
# waveform_16khz = np.zeros(3 * 16000, dtype=np.float32)
# waveform_32khz = np.zeros(5 * 32000, dtype=np.float32)

# # YAMNet
# model = hub.load('https://tfhub.dev/google/yamnet/1')

# # Time inference
# start_time = time.time()
# scores, embeddings, log_mel_spectrogram = model(waveform_16khz)
# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time
# print(f"YAMNet processing time for 5min of audio: {elapsed_time} seconds")

# # Vggish
# model = hub.load('https://www.kaggle.com/models/google/vggish/frameworks/TensorFlow2/variations/vggish/versions/1')

# # Time inference
# start_time = time.time()
# embeddings = model(waveform_16khz)
# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time
# print(f"VGGish processing time for 5min of audio: {elapsed_time} seconds")

# # Perch
# model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4')

# # Time inference
# start_time = time.time()
# logits, embeddings = model.infer_tf(waveform_32khz[np.newaxis, :])
# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time
# print(f"Perch processing time for 5min of audio: {elapsed_time} seconds")

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

class AudioModelInference:
    def __init__(self) -> None:
        # Load YAMNet and VGGish from TensorFlow Hub
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        # Assuming a placeholder load statement for VGGish; replace if you have a direct source.
        self.vggish_model = hub.load('https://tfhub.dev/google/vggish/1')  # Placeholder: Adjust if an actual URL is available.
        # Placeholder for Perch; adjust with actual loading mechanism depending on how you access it
        self.perch_model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4')


    def time_inference(self, model, audio: np.ndarray, model_name: str) -> None:
        start_time = time.time()
        
        if model_name == 'yamnet':
            scores, embeddings, log_mel_spectrogram = model(audio)
        elif model_name == 'vggish':
            # VGGish might need pre-processing similar to YAMNet; adjust based on the actual model usage
            embeddings = model(audio)
        elif model_name == 'perch':
            # Assuming Perch expects a batch dimension and operates on 32 kHz audio
            logits, embeddings = model.infer_tf(audio[np.newaxis, :])
        else:
            raise ValueError("Invalid model name provided.")

        elapsed_time = time.time() - start_time
        print(f"{model_name.capitalize()} processing time for 5 minutes of audio: {elapsed_time} seconds")

    def run(self) -> None:
        # 5 minutes of silence at 16 kHz for YAMNet and VGGish
        five_minutes_16khz = np.zeros(5 * 60 * 16000, dtype=np.float32)

        # 5 minutes of silence at 32 kHz for Perch (assuming it needs 32kHz based on your setup)
        five_minutes_32khz = np.zeros(5 * 60 * 32000, dtype=np.float32)

        # Process with each
        self.time_inference(self.yamnet_model, five_minutes_16khz, 'yamnet')
        self.time_inference(self.vggish_model, five_minutes_16khz, 'vggish')
        self.time_inference(self.perch_model, five_minutes_32khz, 'perch')

# Usage
inference = AudioModelInference()
inference.run()



