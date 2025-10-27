import os
import json
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(output_path, "..", "datasets/processed/Della_3a452b8a/"))

# Pastikan metadata.csv dan file audio sudah benar
metadata_file = os.path.join(dataset_path, "metadata.csv")
assert os.path.isfile(metadata_file), f"metadata.csv tidak ditemukan di {dataset_path}"

# Create speakers.json for multi-speaker setup
speakers_file = os.path.join(dataset_path, "speakers.json")
speakers_data = {"Della_3a452b8a": 0}
with open(speakers_file, 'w') as f:
    json.dump(speakers_data, f, indent=2)

# Multi-speaker dataset configuration
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    path=dataset_path,
    language="id"
)

# VITS Multi-speaker configuration
model_args = VitsArgs(
    use_speaker_embedding=True,
    num_speakers=len(speakers_data),
    speaker_embedding_dim=256,
    use_d_vector_file=False,
    d_vector_dim=0
)

config = VitsConfig(
    model_args=model_args,
    batch_size=8,
    eval_batch_size=4,
    num_loader_workers=2,
    num_epochs=200,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    phoneme_language="id",
    datasets=[dataset_config],
    output_path=output_path,
    audio={
        "sample_rate": 22050,
        "hop_length": 256,
        "win_length": 1024,
        "fft_size": 1024,
        "mel_fmin": 0,
        "mel_fmax": None,
        "num_mels": 80,
        "preemphasis": 0.97,
        "ref_level_db": 20,
        "log_func": "np.log10",
        "do_trim_silence": True,
        "trim_db": 45,
        "power": 1.5,
        "griffin_lim_iters": 60,
    },
    lr=1e-4,
    weight_decay=1e-6,
    grad_clip=5.0,
    lr_scheduler="ExponentialLR",
    lr_scheduler_params={"gamma": 0.999875, "last_epoch": -1},
    use_speaker_embedding=True,
    speakers_file=speakers_file,
    language="id"
)

# Inisialisasi audio processor
ap = AudioProcessor.init_from_config(config)

# Memuat data samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=100,
    eval_split_size=0.1,
)

if len(train_samples) == 0:
    raise RuntimeError("Data training kosong! Cek metadata.csv dan file audio.")

print(f"Jumlah data train: {len(train_samples)}")
print(f"Jumlah data eval: {len(eval_samples)}")
print("Contoh data:", train_samples[0])

# Add speaker information to samples
for sample in train_samples + eval_samples:
    sample["speaker_name"] = "Della_3a452b8a"

# Initialize speaker manager
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

print(f"Number of speakers: {speaker_manager.num_speakers}")
print(f"Speaker names: {speaker_manager.name_to_id}")

# Inisialisasi model
model = Vits(config, ap=ap, tokenizer=None, speaker_manager=speaker_manager)

# Inisialisasi trainer
trainer = Trainer(
    TrainerArgs(
        output_path=output_path,
        save_checkpoints=True,
        print_step=10,
        save_step=500,
        eval_step=500,
        mixed_precision=True,
    ),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    audio_processor=ap,
)

# Mulai training!
trainer.fit()