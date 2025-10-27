# 🎤 Multi-Speaker Voice Cloning Setup

Panduan lengkap untuk menggunakan VITS Multi-speaker dan YourTTS dengan speaker embedding.

## ✨ Fitur Baru

- **VITS Multi-Speaker**: Model VITS dengan dukungan multiple speaker
- **YourTTS Integration**: Model pre-trained multi-lingual untuk voice cloning
- **Speaker Embedding**: Teknologi embedding untuk karakteristik suara yang lebih akurat
- **Fine-tuning**: Fine-tune model pre-trained dengan data custom
- **Voice Adaptation**: Adaptasi suara menggunakan reference audio

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python3 install_dependencies.py
```

### 2. Setup Pretrained Models
```bash
python3 setup_pretrained_models.py
```

### 3. Test Setup
```bash
python3 test_multispeaker.py
```

### 4. Run Application
```bash
python3 app.py
```

## 📖 Cara Penggunaan

### Upload & Training
1. Upload file audio (minimal 2 menit untuk hasil terbaik)
2. Sistem akan otomatis menggunakan **MultiSpeakerVoiceCloner**
3. Training menggunakan speaker embedding dan fine-tuning
4. Epochs yang direkomendasikan: 100-200 (tergantung data)

### Generate Speech
1. Pilih voice yang sudah di-train
2. Masukkan text
3. Sistem akan menggunakan:
   - **Trained model** jika tersedia
   - **Pretrained model + speaker adaptation** sebagai fallback

## 🔧 Model Architecture

### VITS Multi-Speaker
- **Speaker Embedding**: 256 dimensi
- **Multi-Speaker Support**: Unlimited speakers
- **Language**: Indonesian (id)
- **Sample Rate**: 22050 Hz

### YourTTS Features
- **Multi-lingual**: Mendukung banyak bahasa
- **Voice Cloning**: Zero-shot dan few-shot learning
- **Speaker Adaptation**: Menggunakan reference audio

## 📁 File Structure

```
AIdubbing/
├── multispeaker_train.py          # Training script untuk multi-speaker
├── multispeaker_voice_cloner.py   # Multi-speaker voice cloner
├── setup_pretrained_models.py     # Download pretrained models
├── test_multispeaker.py          # Test suite
├── install_dependencies.py       # Dependency installer
├── ai_models/
│   └── train.py                  # Updated training script
├── models/                       # Trained models
│   └── [voice_id]/
│       ├── best_model.pth
│       ├── config.json
│       ├── speakers.json
│       └── voice_profile.json
└── datasets/processed/
    └── [voice_id]/
        ├── clips/
        ├── metadata.csv
        └── speakers.json
```

## ⚙️ Configuration

### Training Parameters
```python
# Multi-speaker VITS
model_args = VitsArgs(
    use_speaker_embedding=True,
    num_speakers=1,  # Will be updated automatically
    speaker_embedding_dim=256,
    use_d_vector_file=False
)

# Training config
config = VitsConfig(
    batch_size=8,           # Reduced for fine-tuning
    num_epochs=200,         # More epochs for better quality
    lr=1e-4,               # Lower learning rate
    use_speaker_embedding=True,
    language="id"
)
```

### Audio Processing
```python
audio_config = {
    "sample_rate": 22050,
    "hop_length": 256,
    "win_length": 1024,
    "fft_size": 1024,
    "num_mels": 80,
    "do_trim_silence": True,
    "trim_db": 45
}
```

## 🎯 Best Practices

### Data Preparation
- **Minimum Duration**: 2 menit audio bersih
- **Optimal Duration**: 5-10 menit
- **Audio Quality**: 22kHz, mono, WAV format
- **Content**: Variasi kalimat dan intonasi

### Training Tips
- **Epochs**: 100-200 (tergantung data)
- **Batch Size**: 8 (untuk GPU terbatas)
- **Learning Rate**: 1e-4 (fine-tuning)
- **Monitoring**: Gunakan TensorBoard

### Inference Optimization
- **Speaker Embedding**: Otomatis dari training data
- **Reference Audio**: Gunakan clip terbaik
- **Text Processing**: Bersihkan teks input

## 🔍 Troubleshooting

### Training Issues
```bash
# Memory error
# Reduce batch_size to 4 or 2

# CUDA out of memory
# Use CPU training or reduce model size

# No speakers found
# Check speakers.json file creation
```

### Generation Issues
```bash
# Model not found
# Check if training completed successfully

# Poor quality output
# Increase training epochs or improve data quality

# Speaker mismatch
# Verify speaker embedding consistency
```

### Dependencies Issues
```bash
# PyTorch installation
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# TTS version conflicts
pip install TTS>=0.22.0

# Audio processing errors
pip install librosa soundfile
```

## 📊 Performance Comparison

| Model Type | Quality | Speed | Memory | Use Case |
|------------|---------|-------|---------|----------|
| Simple TTS | Basic | Fast | Low | Quick testing |
| VITS Multi-Speaker | High | Medium | Medium | Production |
| YourTTS | Very High | Slow | High | Best quality |

## 🎵 Audio Quality Tips

### Input Audio
- Gunakan mikrofon berkualitas
- Rekam di ruangan sunyi
- Hindari background noise
- Konsisten volume dan jarak

### Processing
- Automatic silence trimming
- Noise reduction (optional)
- Normalization
- Segmentation optimization

### Output
- 22kHz sample rate
- Mono channel
- WAV format
- Normalized amplitude

## 🚀 Advanced Features

### Custom Speaker Embedding
```python
# Extract speaker embedding
embedding = cloner.extract_speaker_embedding(audio_path)

# Use custom embedding
cloner.generate_with_embedding(text, embedding, output_path)
```

### Multi-Language Support
```python
# YourTTS multi-language
tts.tts_to_file(
    text="Hello world",
    file_path="output.wav",
    language="en",
    speaker_wav="reference.wav"
)
```

### Batch Processing
```python
# Process multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
for i, text in enumerate(texts):
    cloner.generate_speech(voice_id, text, f"output_{i}.wav")
```

## 📝 Notes

- Training membutuhkan waktu 2-6 jam (tergantung data dan hardware)
- GPU sangat direkomendasikan untuk training
- CPU inference cukup cepat untuk production
- Model size: ~100-500MB per voice
- Kualitas terbaik dengan data 10+ menit

## 🔗 Resources

- [Coqui TTS Documentation](https://tts.readthedocs.io/)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [YourTTS Paper](https://arxiv.org/abs/2112.02418)
- [Speaker Embedding Guide](https://tts.readthedocs.io/en/latest/tutorial_for_nervous_beginners.html)