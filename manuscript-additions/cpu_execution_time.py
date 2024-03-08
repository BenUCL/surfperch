import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

# class AudioModelInference:
#     def __init__(self) -> None:
#         self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
#         self.vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
#         self.perch_model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4')

#     def time_inference_yamnet_vggish(self, model, audio: np.ndarray) -> float:
#         start_time = time.time()
#         _ = model(audio)
#         return time.time() - start_time

#     def time_inference_perch(self, audio: np.ndarray, batch_processing: bool = False) -> float:
#         total_time = 0.0
#         segment_length = 5 * 32000  # 5 seconds at 32 kHz
#         segments = np.array_split(audio, len(audio) // segment_length)

#         if not batch_processing:
#             # Process each segment individually
#             for segment in segments:
#                 if segment.shape[0] < segment_length:
#                     continue  # Skip the last segment if it is shorter than expected
#                 start_time = time.time()
#                 logits, embeddings = self.perch_model.infer_tf(segment[np.newaxis, :])
#                 total_time += time.time() - start_time
#         else:
#             # Batch processing
#             segments = [segment for segment in segments if segment.shape[0] == segment_length]
#             batch = np.stack(segments)
#             start_time = time.time()
#             logits, embeddings = self.perch_model.infer_tf(batch)
#             total_time += time.time() - start_time
        
#         return total_time

#     def run(self) -> None:
#         five_minutes_16khz = np.zeros(5 * 60 * 16000, dtype=np.float32)
#         five_minutes_32khz = np.zeros(5 * 60 * 32000, dtype=np.float32)
        
#         yamnet_time = self.time_inference_yamnet_vggish(self.yamnet_model, five_minutes_16khz)
#         print(f"YAMNet processing time for 5 minutes of audio: {yamnet_time} seconds")

#         vggish_time = self.time_inference_yamnet_vggish(self.vggish_model, five_minutes_16khz)
#         print(f"VGGish processing time for 5 minutes of audio: {vggish_time} seconds")

#         perch_time = self.time_inference_perch(five_minutes_32khz, batch_processing=True)
#         print(f"Perch processing time for 5 minutes of audio with batch processing: {perch_time} seconds")

#         perch_time = self.time_inference_perch(five_minutes_32khz, batch_processing=True)
#         print(f"Perch processing time for 5 minutes of audio with batch processing: {perch_time} seconds")

# # Usage
# inference = AudioModelInference()
# inference.run()

import tensorflow as tf
import numpy as np
import time

class AudioModelInference:
    def __init__(self) -> None:
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        # VGGish loading from a placeholder URL, adjust if there's a specific method required
        self.vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
        # Load the Perch model directly from the provided Kaggle URL
        self.perch_model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4')

    def time_inference_yamnet_vggish(self, model, audio: np.ndarray) -> float:
        # YAMNet and VGGish can directly process the entire audio
        start_time = time.time()
        _ = model(audio)
        return time.time() - start_time

    def time_inference_perch(self, model, audio: np.ndarray) -> float:
        # Perch processes audio in 5-second clips
        total_time = 0.0
        segment_length = 5 * 32000  # 5 seconds at 32 kHz
        segments = np.array_split(audio, len(audio) // segment_length)
        
        for segment in segments:
            if segment.shape[0] < segment_length:
                continue  # Skip the last segment if it is shorter than expected
            start_time = time.time()
            logits, embeddings = model.infer_tf(segment[np.newaxis, :])
            total_time += time.time() - start_time
        return total_time

    def run(self) -> None:
        five_minutes_16khz = np.zeros(5 * 60 * 16000, dtype=np.float32)  # For YAMNet and VGGish
        five_minutes_32khz = np.zeros(5 * 60 * 32000, dtype=np.float32)  # For Perch
        
        yamnet_time = self.time_inference_yamnet_vggish(self.yamnet_model, five_minutes_16khz)
        print(f"YAMNet processing time for 5 minutes of audio: {yamnet_time} seconds")

        vggish_time = self.time_inference_yamnet_vggish(self.vggish_model, five_minutes_16khz)
        print(f"VGGish processing time for 5 minutes of audio: {vggish_time} seconds")

        perch_time = self.time_inference_perch(self.perch_model, five_minutes_32khz)
        print(f"Perch processing time for 5 minutes of audio: {perch_time} seconds")

# Usage
inference = AudioModelInference()
inference.run()

# Batch run perch
def time_inference_perch(model, audio: np.ndarray) -> float:
    total_time = 0.0
    segment_length = 5 * 32000  # 5 seconds at 32 kHz
    
    # Split the audio into 5-second segments
    segments = np.array_split(audio, len(audio) // segment_length)
    
    # Remove any segment that isn't the correct length (likely the last one)
    segments = [segment for segment in segments if segment.shape[0] == segment_length]
    
    # Stack the segments into a single numpy array for batch processing
    batch = np.stack(segments)
    
    start_time = time.time()
    # Assume model.infer_tf can handle batch processing
    logits, embeddings = model.infer_tf(batch)
    total_time += time.time() - start_time
    
    return total_time

perch = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4')
waveform_32khz = np.zeros(5 * 32000, dtype=np.float32)

total_time = time_inference_perch(perch, waveform_32khz)
print(f'Perch batch time for 5 min of audio: {total_time}')

### TODO: put perch function in class + add birdnet + add cpue type

