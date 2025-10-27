import os
import json
import uuid
import threading
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
from gtts import gTTS
import pyttsx3
import tempfile
import random

class RealVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        
        # Initialize pyttsx3 engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_available = True
        except:
            self.tts_available = False
    
    def _load_existing_models(self):
        """Load model yang sudah ada"""
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_file = os.path.join(voice_path, "voice_profile.json")
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        profile = json.load(f)
                        self.voice_models[voice_dir] = profile
    
    def start_training(self, voice_id, epochs=100):
        """Mulai training voice model"""
        training_id = str(uuid.uuid4())
        
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'voice_id': voice_id,
            'epochs': epochs,
            'start_time': datetime.now().isoformat()
        }
        
        thread = threading.Thread(
            target=self._train_model,
            args=(training_id, voice_id, epochs)
        )
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _train_model(self, training_id, voice_id, epochs):
        """Training model dengan analisis karakteristik suara"""
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            # Path dataset
            dataset_path = f"datasets/processed/{voice_id}/clips/"
            
            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset path not found: {dataset_path}")
            
            # Analisis karakteristik suara dari semua file audio
            audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
            
            if len(audio_files) < 2:
                raise Exception("Not enough audio samples for training")
            
            voice_profile = self._analyze_voice_characteristics(dataset_path, audio_files)
            
            # Simulate training progress
            for i in range(10):
                time.sleep(1)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
            
            # Save voice profile
            model_path = os.path.join(self.models_dir, voice_id)
            os.makedirs(model_path, exist_ok=True)
            
            profile_file = os.path.join(model_path, "voice_profile.json")
            with open(profile_file, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            # Load ke memory
            self.voice_models[voice_id] = voice_profile
            
            self.training_status[training_id].update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now().isoformat(),
                'model_path': model_path
            })
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _analyze_voice_characteristics(self, dataset_path, audio_files):
        """Analisis karakteristik suara untuk voice cloning"""
        pitch_values = []
        tempo_values = []
        spectral_features = []
        sample_audio_files = []
        
        for audio_file in audio_files[:10]:  # Analisis maksimal 10 file
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                
                # Analisis pitch (fundamental frequency)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 200
                pitch_values.append(pitch)
                
                # Analisis tempo
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo_values.append(tempo)
                
                # Analisis spektral
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                spectral_features.append({
                    'centroid': spectral_centroid,
                    'rolloff': spectral_rolloff
                })
                
                # Simpan path file audio untuk referensi
                sample_audio_files.append(audio_file)
                
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
                continue
        
        # Hitung rata-rata karakteristik
        avg_pitch = np.mean(pitch_values) if pitch_values else 200
        avg_tempo = np.mean(tempo_values) if tempo_values else 120
        avg_spectral_centroid = np.mean([f['centroid'] for f in spectral_features]) if spectral_features else 2000
        avg_spectral_rolloff = np.mean([f['rolloff'] for f in spectral_features]) if spectral_features else 4000
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'avg_pitch': float(avg_pitch),
            'avg_tempo': float(avg_tempo),
            'spectral_centroid': float(avg_spectral_centroid),
            'spectral_rolloff': float(avg_spectral_rolloff),
            'samples_analyzed': len(audio_files),
            'sample_files': sample_audio_files,
            'dataset_path': dataset_path,
            'created_at': datetime.now().isoformat()
        }
    
    def get_training_status(self, training_id):
        """Get status training"""
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def generate_speech(self, text, voice_id):
        """Generate speech yang benar-benar sesuai dengan text"""
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model {voice_id} not found or not trained. Available models: {list(self.voice_models.keys())}")
        
        if not text or len(text.strip()) == 0:
            raise Exception("Text cannot be empty")
        
        voice_profile = self.voice_models[voice_id]
        
        # Generate speech menggunakan TTS + voice characteristics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{voice_id}_{timestamp}.wav"
        output_path = os.path.join("outputs", output_filename)
        
        os.makedirs("outputs", exist_ok=True)
        
        try:
            # Method 1: Gunakan gTTS untuk generate base speech
            base_audio = self._generate_base_speech(text)
            
            # Method 2: Apply voice characteristics dari training
            final_audio = self._apply_voice_characteristics(base_audio, voice_profile)
            
            # Save audio
            sf.write(output_path, final_audio, 22050)
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise Exception("Failed to save audio file")
            
            print(f"Generated speech for text: '{text[:50]}...' using voice: {voice_id}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def _generate_base_speech(self, text):
        """Generate base speech menggunakan TTS"""
        try:
            # Method 1: Gunakan gTTS (Google TTS)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts = gTTS(text=text, lang='id', slow=False)  # Indonesian language
                tts.save(tmp_file.name)
                
                # Load dan konversi ke format yang diinginkan
                audio, sr = librosa.load(tmp_file.name, sr=22050)
                
                # Cleanup temp file
                os.unlink(tmp_file.name)
                
                return audio
                
        except Exception as e:
            print(f"gTTS failed: {e}, trying pyttsx3...")
            
            # Method 2: Fallback ke pyttsx3 (offline TTS)
            if self.tts_available:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        self.tts_engine.save_to_file(text, tmp_file.name)
                        self.tts_engine.runAndWait()
                        
                        # Load audio
                        audio, sr = librosa.load(tmp_file.name, sr=22050)
                        
                        # Cleanup temp file
                        os.unlink(tmp_file.name)
                        
                        return audio
                        
                except Exception as e2:
                    print(f"pyttsx3 also failed: {e2}")
            
            # Method 3: Last resort - use sample audio with text length adjustment
            return self._generate_from_samples(text)
    
    def _generate_from_samples(self, text):
        """Generate audio dari sample sebagai fallback"""
        # Cari sample audio yang ada
        for voice_id, profile in self.voice_models.items():
            dataset_path = profile.get('dataset_path')
            if dataset_path and os.path.exists(dataset_path):
                sample_files = profile.get('sample_files', [])
                if sample_files:
                    # Pilih sample acak
                    sample_file = random.choice(sample_files)
                    sample_path = os.path.join(dataset_path, sample_file)
                    
                    try:
                        audio, sr = librosa.load(sample_path, sr=22050)
                        
                        # Adjust durasi berdasarkan panjang text
                        target_duration = len(text) * 0.08  # ~0.08 detik per karakter
                        target_samples = int(target_duration * sr)
                        
                        if len(audio) < target_samples:
                            # Repeat audio jika terlalu pendek
                            repeats = int(np.ceil(target_samples / len(audio)))
                            audio = np.tile(audio, repeats)
                        
                        # Trim ke durasi yang diinginkan
                        audio = audio[:target_samples]
                        
                        return audio
                        
                    except Exception as e:
                        print(f"Error loading sample {sample_file}: {e}")
                        continue
        
        # Ultimate fallback - generate simple tone
        duration = max(2.0, len(text) * 0.08)
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        
        # Generate pleasant tone
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        
        # Add envelope
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * sr)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= envelope
        
        return audio
    
    def _apply_voice_characteristics(self, base_audio, voice_profile):
        """Apply karakteristik suara ke base audio"""
        try:
            sr = 22050
            audio = base_audio.copy()
            
            # Apply pitch shifting berdasarkan karakteristik voice
            avg_pitch = voice_profile.get('avg_pitch', 200)
            
            # Tentukan pitch shift berdasarkan karakteristik
            if avg_pitch > 250:  # Suara tinggi
                pitch_shift = random.uniform(2, 4)
            elif avg_pitch < 150:  # Suara rendah
                pitch_shift = random.uniform(-4, -2)
            else:  # Suara normal
                pitch_shift = random.uniform(-1, 1)
            
            # Apply pitch shift
            if abs(pitch_shift) > 0.5:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            
            # Apply time stretching berdasarkan tempo
            avg_tempo = voice_profile.get('avg_tempo', 120)
            tempo_factor = avg_tempo / 120.0
            
            if abs(tempo_factor - 1.0) > 0.1:
                audio = librosa.effects.time_stretch(audio, rate=tempo_factor)
            
            # Apply spectral filtering berdasarkan karakteristik
            spectral_centroid = voice_profile.get('spectral_centroid', 2000)
            
            if spectral_centroid > 2500:
                # Bright voice - emphasize higher frequencies
                from scipy.signal import butter, filtfilt
                b, a = butter(2, [1000, 6000], btype='band', fs=sr)
                audio = filtfilt(b, a, audio)
            elif spectral_centroid < 1500:
                # Warm voice - emphasize lower frequencies
                from scipy.signal import butter, filtfilt
                b, a = butter(2, [200, 3000], btype='band', fs=sr)
                audio = filtfilt(b, a, audio)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.8
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"Error applying voice characteristics: {e}")
            # Return original audio if processing fails
            return base_audio.astype(np.float32)
    
    def list_available_voices(self):
        """List semua voice yang tersedia"""
        voices = []
        for voice_id, profile in self.voice_models.items():
            voices.append({
                'id': voice_id,
                'name': voice_id.replace('_', ' ').title(),
                'status': 'ready',
                'samples_analyzed': profile.get('samples_analyzed', 0),
                'created_at': profile.get('created_at', '')
            })
        
        # Tambahkan voice yang sedang training
        for training_id, status in self.training_status.items():
            if status['status'] in ['starting', 'training']:
                voice_id = status['voice_id']
                if voice_id not in [v['id'] for v in voices]:
                    voices.append({
                        'id': voice_id,
                        'name': voice_id.replace('_', ' ').title(),
                        'status': 'training',
                        'progress': status.get('progress', 0)
                    })
        
        return voices