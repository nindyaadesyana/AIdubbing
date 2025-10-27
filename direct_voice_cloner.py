import os
import json
import uuid
import threading
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import random

class DirectVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
    
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
        """Training model - hanya menyimpan referensi ke audio asli"""
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            # Path dataset
            dataset_path = f"datasets/processed/{voice_id}/clips/"
            
            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset path not found: {dataset_path}")
            
            # Cari file audio yang ada
            audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
            
            if len(audio_files) < 1:
                raise Exception("No audio samples found for training")
            
            print(f"Processing {len(audio_files)} audio files...")
            
            # Simulate training progress
            for i in range(10):
                time.sleep(0.5)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
            
            # Simpan informasi audio files
            voice_profile = {
                'voice_id': voice_id,
                'dataset_path': dataset_path,
                'audio_files': audio_files,
                'sample_count': len(audio_files),
                'created_at': datetime.now().isoformat()
            }
            
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
            
            print(f"Voice training completed for {voice_id}")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def get_training_status(self, training_id):
        """Get status training"""
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def generate_speech(self, text, voice_id):
        """Generate speech menggunakan audio asli langsung"""
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
            # Langsung gunakan audio asli
            final_audio = self._use_original_voice_directly(text, voice_profile)
            
            # Save audio
            sf.write(output_path, final_audio, 22050)
            
            if not os.path.exists(output_path):
                raise Exception("Failed to save audio file")
            
            print(f"Generated speech using original voice: '{text[:30]}...'")
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def _use_original_voice_directly(self, text, voice_profile):
        """Gunakan suara asli langsung tanpa modifikasi berlebihan"""
        dataset_path = voice_profile.get('dataset_path')
        audio_files = voice_profile.get('audio_files', [])
        
        if not dataset_path or not audio_files:
            raise Exception("No voice samples available")
        
        # Tentukan berapa banyak audio yang dibutuhkan berdasarkan panjang text
        text_length = len(text)
        
        if text_length <= 30:  # Text pendek - 1 audio
            num_samples = 1
        elif text_length <= 80:  # Text sedang - 2 audio
            num_samples = 2
        else:  # Text panjang - 3 audio
            num_samples = 3
        
        # Pilih audio files secara acak
        selected_files = random.sample(audio_files, min(num_samples, len(audio_files)))
        
        combined_audio = []
        
        for audio_file in selected_files:
            audio_path = os.path.join(dataset_path, audio_file)
            
            try:
                # Load audio asli
                audio, sr = librosa.load(audio_path, sr=22050)
                
                # Hanya trim silence di awal dan akhir
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=15)
                
                if len(audio_trimmed) > 0:
                    combined_audio.append(audio_trimmed)
                    print(f"Using audio: {audio_file} (duration: {len(audio_trimmed)/sr:.1f}s)")
                
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                continue
        
        if not combined_audio:
            raise Exception("No valid audio samples could be loaded")
        
        # Gabungkan audio dengan jeda natural
        if len(combined_audio) == 1:
            final_audio = combined_audio[0]
        else:
            # Jeda antar audio
            silence_duration = 0.2  # 200ms
            silence = np.zeros(int(silence_duration * 22050))
            
            final_audio = combined_audio[0]
            for audio_segment in combined_audio[1:]:
                final_audio = np.concatenate([final_audio, silence, audio_segment])
        
        # Adjust durasi berdasarkan text (opsional)
        target_duration = max(len(final_audio) / 22050, len(text) * 0.06)  # minimal 0.06 detik per karakter
        target_samples = int(target_duration * 22050)
        
        if len(final_audio) < target_samples:
            # Extend dengan mengulangi audio terakhir
            last_audio = combined_audio[-1]
            gap = np.zeros(int(0.3 * 22050))  # 300ms gap
            
            while len(final_audio) < target_samples:
                remaining_samples = target_samples - len(final_audio)
                if remaining_samples > len(gap) + len(last_audio):
                    final_audio = np.concatenate([final_audio, gap, last_audio])
                else:
                    # Tambahkan sisa yang dibutuhkan
                    final_audio = np.concatenate([final_audio, gap[:remaining_samples]])
                    break
        
        # Trim jika terlalu panjang
        max_duration = max(10.0, len(text) * 0.15)  # maksimal 0.15 detik per karakter
        max_samples = int(max_duration * 22050)
        if len(final_audio) > max_samples:
            final_audio = final_audio[:max_samples]
        
        # Normalisasi volume
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val * 0.9  # 90% volume untuk headroom
        
        # Fade in/out yang halus
        fade_samples = int(0.05 * 22050)  # 50ms fade
        if len(final_audio) > fade_samples * 2:
            # Fade in
            final_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade out
            final_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        print(f"Final audio duration: {len(final_audio)/22050:.1f} seconds")
        
        return final_audio.astype(np.float32)
    
    def list_available_voices(self):
        """List semua voice yang tersedia"""
        voices = []
        for voice_id, profile in self.voice_models.items():
            voices.append({
                'id': voice_id,
                'name': voice_id.replace('_', ' ').title(),
                'status': 'ready',
                'samples_analyzed': profile.get('sample_count', 0),
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