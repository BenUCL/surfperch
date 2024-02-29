 #@title Imports. { vertical-output: true }
from etils import epath
from ml_collections import config_dict
import numpy as np
import os
import tensorflow as tf
import tqdm
from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

from chirp import audio_utils
from chirp.inference import embed_lib
from chirp.inference import tf_examples



#@title Configuration. { vertical-output: true }
base_dir = os.getenv('BASE_DIR')
if not base_dir:
    raise ValueError("BASE_DIR environment variable is not set.")

# Define the model
model_choice = 'perch'  #@param

config = config_dict.ConfigDict()
config.embed_fn_config = config_dict.ConfigDict()
config.embed_fn_config.model_config = config_dict.ConfigDict()

# Pick the input and output targets.
# source_file_patterns should contain a list of globs of audio files, like:
# ['/home/me/*.wav','/home/me/*.WAV', '/home/me/other/*.flac']
config.source_file_patterns = [os.path.join(base_dir,'ucl_perch/data/australia/raw_audio/*.WAV')] #@param
config.output_dir = os.path.join(base_dir,'ucl_perch/data/australia/embeddings')  #@param

# For Perch, the directory containing the model.
# Alternatively, set the perch_tfhub_model_version, and the model will load
# directly from TFHub.
# Note that only one of perch_model_path and perch_tfhub_version should be set.
perch_model_path = os.path.join(base_dir,'ucl_perch/model/reef_perch_model/')  #@param
#perch_tfhub_version = 4  #@param

# For BirdNET, point to the specific tflite file.
birdnet_model_path = ''  #@param
if model_choice == 'perch':
  config.embed_fn_config.model_key = 'taxonomy_model_tf'
  config.embed_fn_config.model_config.window_size_s = 5.0
  config.embed_fn_config.model_config.hop_size_s = 5.0
  config.embed_fn_config.model_config.sample_rate = 32000
  #config.embed_fn_config.model_config.tfhub_version = perch_tfhub_version
  config.embed_fn_config.model_config.model_path = perch_model_path
elif model_choice == 'birdnet':
  config.embed_fn_config.model_key = 'birdnet'
  config.embed_fn_config.model_config.window_size_s = 3.0
  config.embed_fn_config.model_config.hop_size_s = 3.0
  config.embed_fn_config.model_config.sample_rate = 48000
  config.embed_fn_config.model_config.model_path = birdnet_model_path
  # Note: This class list is appropriate for Birdnet 2.1, 2.2, and 2.3
  config.embed_fn_config.model_config.class_list_name = 'birdnet_v2_1'
  config.embed_fn_config.model_config.num_tflite_threads = 4

# Only write embeddings to reduce size.
config.embed_fn_config.write_embeddings = True
config.embed_fn_config.write_logits = False
config.embed_fn_config.write_separated_audio = False
config.embed_fn_config.write_raw_audio = False

# Number of parent directories to include in the filename.
config.embed_fn_config.file_id_depth = 1






#@title Set up. { vertical-output: true }

# Set up the embedding function, including loading models.
embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
print('\n\nLoading model(s)...')
embed_fn.setup()

# Create output directory and write the configuration.
output_dir = epath.Path(config.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
embed_lib.maybe_write_config(config, output_dir)

# Create SourceInfos.
source_infos = embed_lib.create_source_infos(
    config.source_file_patterns,
    num_shards_per_file=config.get('num_shards_per_file', -1),
    shard_len_s=config.get('shard_len_s', -1))
print(f'Found {len(source_infos)} source infos.')

print('\n\nTest-run of model...')
window_size_s = config.embed_fn_config.model_config.window_size_s
sr = config.embed_fn_config.model_config.sample_rate
z = np.zeros([int(sr * window_size_s)])
embed_fn.embedding_model.embed(z)
print('Setup complete!')






#@title Run embedding. { vertical-output: true }

# Uses multiple threads to load audio before embedding.
# This tends to be faster, but can fail if any audio files are corrupt.

embed_fn.min_audio_s = 1.0
record_file = (output_dir / 'embeddings.tfrecord').as_posix()
succ, fail = 0, 0

existing_embedding_ids = embed_lib.get_existing_source_ids(
    output_dir, 'embeddings-*')

source_infos = embed_lib.get_new_source_infos(
    source_infos, existing_embedding_ids, config.embed_fn_config.file_id_depth)

print(f'Processing {len(source_infos)} new source infos.')

audio_iterator = audio_utils.multi_load_audio_window(
    filepaths=[s.filepath for s in source_infos],
    offsets=[s.shard_num * s.shard_len_s for s in source_infos],
    sample_rate=config.embed_fn_config.model_config.sample_rate,
    window_size_s=config.get('shard_len_s', -1.0),
)
with tf_examples.EmbeddingsTFRecordMultiWriter(
    output_dir=output_dir, num_files=config.get('tf_record_shards', 1)) as file_writer:
  for source_info, audio in tqdm.tqdm(
      zip(source_infos, audio_iterator), total=len(source_infos)):
    file_id = source_info.file_id(config.embed_fn_config.file_id_depth)
    offset_s = source_info.shard_num * source_info.shard_len_s
    example = embed_fn.audio_to_example(file_id, offset_s, audio)
    if example is None:
      fail += 1
      continue
    file_writer.write(example.SerializeToString())
    succ += 1
  file_writer.flush()
print(f'\n\nSuccessfully processed {succ} source_infos, failed {fail} times.')

fns = [fn for fn in output_dir.glob('embeddings-*')]
ds = tf.data.TFRecordDataset(fns)
parser = tf_examples.get_example_parser()
ds = ds.map(parser)
for ex in ds.as_numpy_iterator():
  print(ex['filename'])
  print(ex['embedding'].shape)
  break
