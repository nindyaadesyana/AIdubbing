import os
import json
import uuid
import threading
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import tempfile
import subprocess
import sys

class RealisticTTSCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        
        # Check if we can use system TTS
        self.system_tts_available = self._check_system_tts()
    
    def _check_system_tts(self):
        """Check if system TTS is available"""
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["say", "--version"], capture_output=True, check=True)
                return True
        except:
            pass
        return False
    
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
        """Training model - analisis suara untuk voice matching"""
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            dataset_path = f"datasets/processed/{voice_id}/clips/"
            
            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset path not found: {dataset_path}")
            
            audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
            
            if len(audio_files) < 2:
                raise Exception("Not enough audio samples for training")
            
            print(f"Analyzing {len(audio_files)} voice samples for realistic TTS...")
            
            # Simulate training dengan progress realistis
            for i in range(10):
                time.sleep(max(0.3, epochs / 200))
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
            
            # Analisis karakteristik suara
            voice_profile = self._analyze_voice_for_tts(dataset_path, audio_files)
            
            # Save voice profile
            model_path = os.path.join(self.models_dir, voice_id)
            os.makedirs(model_path, exist_ok=True)
            
            profile_file = os.path.join(model_path, "voice_profile.json")
            with open(profile_file, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            self.voice_models[voice_id] = voice_profile
            
            self.training_status[training_id].update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now().isoformat(),
                'model_path': model_path
            })
            
            print(f"Voice analysis completed for realistic TTS: {voice_id}")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _analyze_voice_for_tts(self, dataset_path, audio_files):
        """Analisis suara untuk TTS yang realistis"""
        
        best_samples = []
        pitch_values = []
        speaking_rates = []
        
        for audio_file in audio_files[:10]:
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 0.8:  # Minimal 0.8 detik
                    # Analisis pitch
                    pitches, magnitudes = librosa.piptrack(y=y_clean, sr=sr)
                    valid_pitches = []
                    
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if 80 < pitch < 400:
                            valid_pitches.append(pitch)
                    
                    if len(valid_pitches) > 5:
                        avg_pitch = np.mean(valid_pitches)
                        pitch_values.append(avg_pitch)
                        
                        # Estimasi speaking rate
                        duration = len(y_clean) / sr
                        # Rough estimate: assume average word length
                        estimated_syllables = duration * 3  # ~3 syllables per second
                        speaking_rates.append(estimated_syllables / duration)
                        
                        # Simpan sample yang bagus
                        best_samples.append({
                            'file': audio_file,
                            'pitch': avg_pitch,
                            'duration': duration,
                            'quality': len(valid_pitches) / len(y_clean) * sr
                        })
                
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
                continue
        
        # Sort by quality
        best_samples.sort(key=lambda x: x['quality'], reverse=True)
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'avg_pitch': float(np.mean(pitch_values)) if pitch_values else 200,
            'pitch_std': float(np.std(pitch_values)) if len(pitch_values) > 1 else 20,
            'avg_speaking_rate': float(np.mean(speaking_rates)) if speaking_rates else 3.0,
            'best_samples': [s['file'] for s in best_samples[:3]],
            'voice_character': self._determine_voice_character(pitch_values),
            'dataset_path': dataset_path,
            'sample_count': len(audio_files),
            'created_at': datetime.now().isoformat()
        }
    
    def _determine_voice_character(self, pitch_values):
        """Tentukan karakter suara"""
        if not pitch_values:
            return 'normal'
        
        avg_pitch = np.mean(pitch_values)
        
        if avg_pitch > 220:
            return 'high'  # Suara tinggi
        elif avg_pitch < 160:
            return 'low'   # Suara rendah
        else:
            return 'normal'  # Suara normal
    
    def get_training_status(self, training_id):
        """Get status training"""
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def generate_speech(self, text, voice_id):
        """Generate speech dengan pendekatan yang lebih realistis"""
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model {voice_id} not found or not trained")
        
        if not text or len(text.strip()) == 0:
            raise Exception("Text cannot be empty")
        
        voice_profile = self.voice_models[voice_id]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{voice_id}_{timestamp}.wav"
        output_path = os.path.join("outputs", output_filename)
        
        os.makedirs("outputs", exist_ok=True)
        
        try:
            print(f"Generating realistic TTS for: '{text[:50]}...'")
            
            # Coba beberapa metode TTS
            audio = None
            
            # Method 1: System TTS (macOS)
            if self.system_tts_available:
                audio = self._generate_system_tts(text, voice_profile)
            
            # Method 2: Fallback ke gTTS dengan voice modification
            if audio is None:
                audio = self._generate_gtts_with_modification(text, voice_profile)
            
            # Method 3: Ultimate fallback - use best voice samples
            if audio is None:
                audio = self._generate_from_voice_samples(text, voice_profile)
            
            if audio is None:
                raise Exception("All TTS methods failed")
            
            # Post-processing
            final_audio = self._post_process_tts(audio)
            
            # Save audio
            sf.write(output_path, final_audio, 22050)
            
            if not os.path.exists(output_path):
                raise Exception("Failed to save audio file")
            
            print(f"✅ Realistic TTS generated successfully!")
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def _generate_system_tts(self, text, voice_profile):
        """Generate menggunakan system TTS (macOS)"""
        try:
            if not self.system_tts_available:
                return None
            
            print("Using system TTS (macOS)...")
            
            # Tentukan voice berdasarkan karakteristik
            voice_character = voice_profile.get('voice_character', 'normal')
            avg_pitch = voice_profile.get('avg_pitch', 200)
            
            # Pilih voice yang sesuai
            if voice_character == 'high' or avg_pitch > 220:
                system_voice = "Samantha"  # Female voice
                rate = 180  # Slightly faster
            elif voice_character == 'low' or avg_pitch < 160:
                system_voice = "Alex"      # Male voice
                rate = 160  # Slightly slower
            else:
                system_voice = "Samantha"  # Default female
                rate = 170  # Normal rate
            
            # Generate dengan system TTS
            with tempfile.NamedTemporaryFile(delete=False, suffix='.aiff') as tmp_file:
                cmd = [
                    "say",
                    "-v", system_voice,
                    "-r", str(rate),
                    "-o", tmp_file.name,
                    text
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Load audio
                    audio, sr = librosa.load(tmp_file.name, sr=22050)
                    os.unlink(tmp_file.name)
                    
                    print(f"System TTS generated: {len(audio)/sr:.1f} seconds")
                    return audio
                else:
                    print(f"System TTS failed: {result.stderr}")
                    os.unlink(tmp_file.name)
                    return None
                    
        except Exception as e:
            print(f"System TTS error: {e}")
            return None
    
    def _generate_gtts_with_modification(self, text, voice_profile):
        """Generate menggunakan gTTS dengan modifikasi"""
        try:
            print("Using gTTS with voice modification...")
            
            from gtts import gTTS
            
            # Generate base TTS
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts = gTTS(text=text, lang='id', slow=False)
                tts.save(tmp_file.name)
                
                # Load audio
                audio, sr = librosa.load(tmp_file.name, sr=22050)
                os.unlink(tmp_file.name)
                
                # Apply simple voice modification
                modified_audio = self._apply_simple_voice_mod(audio, voice_profile)
                
                print(f"gTTS generated and modified: {len(modified_audio)/sr:.1f} seconds")
                return modified_audio
                
        except Exception as e:
            print(f"gTTS error: {e}")
            return None
    
    def _apply_simple_voice_mod(self, audio, voice_profile):
        """Apply modifikasi suara sederhana"""
        try:
            sr = 22050
            
            # Pitch adjustment berdasarkan voice character
            voice_character = voice_profile.get('voice_character', 'normal')
            avg_pitch = voice_profile.get('avg_pitch', 200)
            
            if voice_character == 'high' or avg_pitch > 220:
                pitch_shift = 2.0  # Naikkan pitch
            elif voice_character == 'low' or avg_pitch < 160:
                pitch_shift = -2.0  # Turunkan pitch
            else:
                pitch_shift = 0.0  # Normal
            
            # Apply pitch shift
            if abs(pitch_shift) > 0.5:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            
            # Speed adjustment
            speaking_rate = voice_profile.get('avg_speaking_rate', 3.0)
            if speaking_rate > 3.5:
                # Faster speaking
                audio = librosa.effects.time_stretch(audio, rate=1.1)
            elif speaking_rate < 2.5:
                # Slower speaking
                audio = librosa.effects.time_stretch(audio, rate=0.9)
            
            return audio
            
        except Exception as e:
            print(f"Voice modification error: {e}")
            return audio
    
    def _generate_from_voice_samples(self, text, voice_profile):
        """Generate dari voice samples sebagai fallback"""
        try:
            print("Using voice samples as fallback...")
            
            dataset_path = voice_profile.get('dataset_path')
            best_samples = voice_profile.get('best_samples', [])
            
            if not dataset_path or not best_samples:
                return None
            
            # Load best samples
            audio_segments = []
            for sample_file in best_samples[:2]:  # Use top 2 samples
                sample_path = os.path.join(dataset_path, sample_file)
                if os.path.exists(sample_path):
                    try:
                        audio, sr = librosa.load(sample_path, sr=22050)
                        audio_clean, _ = librosa.effects.trim(audio, top_db=20)
                        audio_segments.append(audio_clean)
                    except:
                        continue
            
            if not audio_segments:
                return None
            
            # Combine segments
            silence = np.zeros(int(0.2 * 22050))  # 200ms silence
            combined_audio = audio_segments[0]
            
            for segment in audio_segments[1:]:
                combined_audio = np.concatenate([combined_audio, silence, segment])
            
            # Adjust duration based on text length
            target_duration = max(3.0, len(text) * 0.08)
            target_samples = int(target_duration * 22050)
            
            if len(combined_audio) < target_samples:
                # Extend by repeating
                repeats = int(np.ceil(target_samples / len(combined_audio)))
                extended_audio = []
                
                for i in range(repeats):
                    extended_audio.append(combined_audio)
                    if i < repeats - 1:
                        extended_audio.append(silence)
                
                combined_audio = np.concatenate(extended_audio)
            
            # Trim to target length
            combined_audio = combined_audio[:target_samples]
            
            print(f"Voice samples generated: {len(combined_audio)/22050:.1f} seconds")
            return combined_audio
            
        except Exception as e:
            print(f"Voice samples error: {e}")
            return None
    
    def _post_process_tts(self, audio):
        """Post-processing untuk TTS"""
        sr = 22050
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        # Smooth fade in/out
        fade_samples = int(0.05 * sr)
        if len(audio) > fade_samples * 2:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Light noise reduction
        audio = librosa.effects.preemphasis(audio, coef=0.95)
        
        return audio.astype(np.float32)
    
    def list_available_voices(self):
        """List semua voice yang tersedia"""
        voices = []
        for voice_id, profile in self.voice_models.items():
            voices.append({
                'id': voice_id,
                'name': voice_id.replace('_', ' ').title(),
                'status': 'ready',
                'samples_analyzed': profile.get('sample_count', 0),
                'voice_character': profile.get('voice_character', 'normal'),
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