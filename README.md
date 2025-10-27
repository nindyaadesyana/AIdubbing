# 🎤 AI Voice Cloning - Dubbing App

Aplikasi web untuk voice cloning yang dapat mengubah file MP3 menjadi model suara untuk dubbing.

## ✨ Fitur

- **Upload Audio**: Upload file audio (MP3, M4A, WAV, FLAC, AAC, OGG, WMA) untuk dijadikan sample voice
- **Auto Processing**: Otomatis memotong audio berdasarkan silence
- **Voice Training**: Training model VITS untuk voice cloning
- **Speech Generation**: Generate speech dari text menggunakan voice yang sudah di-train
- **Web Interface**: Interface web yang user-friendly

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```bash
python run.py
```

### 3. Buka Browser
Akses aplikasi di: `http://localhost:5000`

## 📖 Cara Penggunaan

### Upload Voice
1. Masuk ke tab "Upload Voice"
2. Masukkan nama voice (contoh: "Della")
3. Upload file audio (minimal 30 detik) - mendukung MP3, M4A, WAV, FLAC, AAC, OGG, WMA
4. Klik "Upload & Process"

### Training Model
1. Setelah upload berhasil, atur jumlah epochs (default: 100)
2. Klik "Start Training"
3. Tunggu proses training selesai (bisa 30 menit - 2 jam)

### Generate Speech
1. Masuk ke tab "Generate Speech"
2. Pilih voice yang sudah di-train
3. Masukkan text yang ingin di-dubbing
4. Klik "Generate Speech"
5. Download hasil audio

## 📁 Struktur Project

```
AIdubbing/
├── app.py              # Main Flask application
├── voice_cloner.py     # Voice cloning logic
├── audio_processor.py  # Audio processing utilities
├── run.py             # Application launcher
├── requirements.txt   # Dependencies
├── templates/         # HTML templates
├── static/           # CSS & JS files
├── uploads/          # Uploaded audio files
├── models/           # Trained voice models
├── outputs/          # Generated audio files
└── datasets/         # Processed training data
```

## ⚙️ Konfigurasi

- **Sample Rate**: 22050 Hz
- **Audio Format**: WAV, Mono
- **Min Duration**: 1 detik per segment
- **Max Duration**: 10 detik per segment
- **Model**: VITS (Variational Inference TTS)

## 🔧 Troubleshooting

### Error saat install dependencies
```bash
# Untuk macOS dengan Apple Silicon
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Untuk Linux/Windows
pip install torch torchaudio
```

### Audio terlalu pendek atau tidak cukup segmen
- Gunakan audio minimal 10 detik, idealnya 30 detik atau lebih
- Pastikan audio memiliki jeda/silence yang jelas untuk segmentasi otomatis
- Jika audio tanpa jeda, sistem akan membagi otomatis menjadi segmen 3 detik
- Hindari audio yang terlalu monoton atau tanpa variasi

### Audio tidak terpotong dengan baik
- Pastikan file audio memiliki jeda/silence yang jelas
- Gunakan audio dengan kualitas baik (minimal 22kHz)
- Hindari background noise yang terlalu keras
- Coba kurangi volume background music jika ada

### Training gagal
- Pastikan ada minimal 2 segment audio yang valid
- Check apakah ada cukup disk space (minimal 1GB)
- Reduce jumlah epochs jika memory terbatas
- Pastikan audio tidak corrupt atau rusak

## 📝 Notes

- Training membutuhkan waktu lama, bersabar
- Kualitas hasil bergantung pada kualitas input MP3
- Untuk hasil terbaik, gunakan audio yang jelas dan bersih