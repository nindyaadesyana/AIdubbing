import os
import uuid
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import os

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.min_duration = 1.0  # minimum 1 detik
        self.max_duration = 10.0  # maksimum 10 detik
    
    def analyze_audio(self, audio_path):
        """Analisis audio untuk memberikan info kepada user"""
        try:
            audio = self._load_audio_file(audio_path)
            duration = len(audio) / 1000.0  # durasi dalam detik
            
            # Coba split untuk estimasi segmen
            chunks = split_on_silence(
                audio,
                min_silence_len=300,
                silence_thresh=audio.dBFS - 20,
                keep_silence=100
            )
            
            return {
                'duration': duration,
                'estimated_segments': len(chunks),
                'audio_quality': 'good' if audio.dBFS > -30 else 'low',
                'sample_rate': audio.frame_rate
            }
        except Exception as e:
            return {'error': str(e)}
    
    def process_for_training(self, audio_path, voice_name):
        """Proses file audio untuk training"""
        voice_id = f"{voice_name}_{str(uuid.uuid4())[:8]}"
        
        # Analisis audio terlebih dahulu
        analysis = self.analyze_audio(audio_path)
        if 'error' in analysis:
            raise Exception(f"Error analyzing audio: {analysis['error']}")
        
        if analysis['duration'] < 5:
            raise Exception(f"Audio terlalu pendek ({analysis['duration']:.1f} detik). Minimal 5 detik diperlukan.")
        
        # Buat direktori output
        output_dir = f"datasets/processed/{voice_id}"
        clips_dir = os.path.join(output_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        
        # Load audio berdasarkan format
        audio = self._load_audio_file(audio_path)
        
        # Normalize audio
        audio = audio.normalize()
        
        # Split berdasarkan silence dengan parameter yang lebih fleksibel
        chunks = split_on_silence(
            audio,
            min_silence_len=300,  # 300ms silence (lebih pendek)
            silence_thresh=audio.dBFS - 20,  # threshold lebih sensitif
            keep_silence=100  # keep 100ms silence
        )
        
        # Jika tidak ada chunks, bagi audio menjadi segmen tetap
        if len(chunks) < 3:
            chunk_length = 3000  # 3 detik per chunk
            chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
        
        # Filter chunks berdasarkan durasi dengan parameter lebih fleksibel
        valid_chunks = []
        for chunk in chunks:
            duration = len(chunk) / 1000.0  # durasi dalam detik
            if 0.5 <= duration <= self.max_duration:  # minimal 0.5 detik
                valid_chunks.append(chunk)
        
        # Jika masih kurang, bagi chunk panjang menjadi beberapa bagian
        if len(valid_chunks) < 3:
            extended_chunks = []
            for chunk in chunks:
                duration = len(chunk) / 1000.0
                if duration > self.max_duration:
                    # Bagi chunk panjang
                    segment_length = int(self.max_duration * 1000)  # dalam ms
                    for i in range(0, len(chunk), segment_length):
                        sub_chunk = chunk[i:i+segment_length]
                        if len(sub_chunk) >= 500:  # minimal 0.5 detik
                            extended_chunks.append(sub_chunk)
                else:
                    extended_chunks.append(chunk)
            valid_chunks = extended_chunks
        
        if len(valid_chunks) < 2:
            raise Exception(f"Audio terlalu pendek. Ditemukan {len(valid_chunks)} segmen, minimal butuh 2 segmen. Coba gunakan audio yang lebih panjang (minimal 10 detik).")
        
        # Simpan chunks dan buat metadata
        metadata_lines = []
        
        for i, chunk in enumerate(valid_chunks[:30]):  # maksimal 30 chunks
            # Export ke WAV
            filename = f"{voice_id}_{i+1:03d}.wav"
            filepath = os.path.join(clips_dir, filename)
            
            # Konversi ke format yang dibutuhkan
            chunk = chunk.set_frame_rate(self.sample_rate)
            chunk = chunk.set_channels(1)  # mono
            chunk.export(filepath, format="wav")
            
            # Generate text placeholder (bisa diganti dengan speech-to-text)
            text = self._generate_placeholder_text(i)
            metadata_lines.append(f"{filename}|{text}")
        
        # Simpan metadata.csv
        metadata_path = os.path.join(output_dir, "metadata.csv")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))
        
        return {
            'voice_id': voice_id,
            'samples_count': len(metadata_lines),
            'dataset_path': output_dir,
            'audio_duration': analysis['duration'],
            'audio_quality': analysis['audio_quality']
        }
    
    def _generate_placeholder_text(self, index):
        """Generate placeholder text untuk training"""
        # Dalam implementasi nyata, gunakan speech-to-text
        placeholder_texts = [
            "Halo, ini adalah rekaman suara untuk voice cloning.",
            "Saya sedang berbicara dengan jelas dan natural.",
            "Teknologi voice cloning sangat menarik untuk dipelajari.",
            "Kualitas audio yang baik sangat penting untuk hasil terbaik.",
            "Mari kita coba membuat suara yang mirip dengan aslinya.",
            "Proses training membutuhkan data audio yang berkualitas.",
            "Setiap kata harus diucapkan dengan artikulasi yang jelas.",
            "Voice cloning dapat digunakan untuk berbagai aplikasi.",
            "Penting untuk memiliki variasi intonasi dalam rekaman.",
            "Hasil akhir akan bergantung pada kualitas data training."
        ]
        
        return placeholder_texts[index % len(placeholder_texts)]
    
    def _load_audio_file(self, audio_path):
        """Load audio file dengan berbagai format"""
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        try:
            if file_ext == '.mp3':
                audio = AudioSegment.from_mp3(audio_path)
            elif file_ext == '.m4a':
                audio = AudioSegment.from_file(audio_path, format="m4a")
            elif file_ext == '.wav':
                audio = AudioSegment.from_wav(audio_path)
            elif file_ext == '.flac':
                audio = AudioSegment.from_file(audio_path, format="flac")
            elif file_ext == '.aac':
                audio = AudioSegment.from_file(audio_path, format="aac")
            elif file_ext == '.ogg':
                audio = AudioSegment.from_ogg(audio_path)
            elif file_ext == '.wma':
                audio = AudioSegment.from_file(audio_path, format="wma")
            else:
                # Fallback: coba load dengan librosa
                audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
                # Konversi ke AudioSegment
                audio_data = (audio_data * 32767).astype(np.int16)
                audio = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sr,
                    sample_width=2,
                    channels=1
                )
            
            return audio
            
        except Exception as e:
            raise Exception(f"Error loading audio file {audio_path}: {str(e)}")
    
    def convert_audio_to_wav(self, audio_path, output_path=None):
        """Konversi berbagai format audio ke WAV"""
        if output_path is None:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}.wav"
        
        # Load dengan librosa untuk konsistensi
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Simpan sebagai WAV
        sf.write(output_path, audio, sr)
        
        return output_path
    
    def enhance_audio(self, audio_path):
        """Enhance kualitas audio"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Noise reduction sederhana
        audio = librosa.effects.preemphasis(audio)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        return audio, sr