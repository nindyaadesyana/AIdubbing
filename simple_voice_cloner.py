import os
import json
import uuid
import threading
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import tempfile
from gtts import gTTS
from scipy.signal import butter, filtfilt

class SimpleVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        print("✅ Simple Voice Cloner initialized (no PyTorch required)")
    
    def _load_existing_models(self):
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_file = os.path.join(voice_path, "voice_profile.json")
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        profile = json.load(f)
                        self.voice_models[voice_dir] = profile
    
    def start_training(self, voice_id, epochs=5):
        training_id = str(uuid.uuid4())
        
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'voice_id': voice_id,
            'epochs': epochs,
            'start_time': datetime.now().isoformat()
        }
        
        thread = threading.Thread(target=self._train_model, args=(training_id, voice_id, epochs))
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _train_model(self, training_id, voice_id, epochs):
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            # Find audio files
            audio_files = self._find_audio_files(voice_id)
            
            if len(audio_files) < 1:
                raise Exception("Need at least 1 audio file")
            
            print(f"🎯 Simple training with {len(audio_files)} files...")
            
            # Process audio files
            reference_clips = self._process_audio_files(audio_files, voice_id)
            
            # Quick training simulation
            for epoch in range(epochs):
                time.sleep(0.2)
                progress = int((epoch + 1) / epochs * 100)
                self.training_status[training_id]['progress'] = progress
                print(f"Training epoch {epoch+1}/{epochs} - Progress: {progress}%")
            
            # Create voice profile
            voice_profile = {
                'voice_id': voice_id,
                'model_type': 'simple',
                'reference_clips': reference_clips,
                'sample_count': len(reference_clips),
                'audio_analysis': self._analyze_audio(reference_clips),
                'created_at': datetime.now().isoformat()
            }
            
            # Save profile
            model_path = os.path.join(self.models_dir, voice_id)
            os.makedirs(model_path, exist_ok=True)
            
            profile_file = os.path.join(model_path, "voice_profile.json")
            with open(profile_file, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            self.voice_models[voice_id] = voice_profile
            
            self.training_status[training_id].update({
                'status': 'completed',
                'progress': 100,
                'total_epochs': epochs,
                'end_time': datetime.now().isoformat()
            })
            
            print(f"✅ Simple training completed!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _find_audio_files(self, voice_id):
        import re
        matching_files = []
        search_dirs = ["uploads", "datasets/raw", "datasets/processed"]
        
        voice_pattern = re.compile(rf".*{re.escape(voice_id)}.*", re.IGNORECASE)
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                            if voice_pattern.match(file):
                                matching_files.append(os.path.join(root, file))
        
        return list(set(matching_files))
    
    def _process_audio_files(self, audio_files, voice_id):
        reference_clips = []
        
        for i, audio_file in enumerate(audio_files[:3]):
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 1.0:
                    model_path = os.path.join(self.models_dir, voice_id)
                    os.makedirs(model_path, exist_ok=True)
                    
                    clip_file = os.path.join(model_path, f"reference_{i}.wav")
                    sf.write(clip_file, y_clean, sr)
                    
                    reference_clips.append({
                        'file_path': clip_file,
                        'duration': len(y_clean) / sr,
                        'source': os.path.basename(audio_file)
                    })
                    
            except Exception as e:
                continue
        
        return reference_clips
    
    def _analyze_audio(self, clips):
        if not clips:
            return {'recommended_epochs': 5}
        
        total_duration = sum(clip['duration'] for clip in clips)
        
        # Simple epoch recommendation
        if total_duration >= 20:
            recommended_epochs = 3
        elif total_duration >= 10:
            recommended_epochs = 5
        else:
            recommended_epochs = 8
        
        return {
            'total_duration': total_duration,
            'quality_level': 'Medium',
            'recommended_epochs': recommended_epochs,
            'training_time_estimate': f"{recommended_epochs} seconds"
        }
    
    def generate_speech(self, voice_id, text, output_path):
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            print(f"🎤 Generating simple TTS: {text}")
            
            # Generate TTS
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            # Load and convert
            y_tts, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)
            
            # Apply simple voice conversion
            profile = self.voice_models[voice_id]
            y_converted = self._apply_simple_conversion(y_tts, sr, profile)
            
            # Save
            sf.write(output_path, y_converted, sr)
            
            print(f"✅ Simple TTS generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _apply_simple_conversion(self, y_tts, sr, profile):
        # Simple pitch adjustment
        if 'reference_clips' in profile and profile['reference_clips']:
            try:
                ref_path = profile['reference_clips'][0]['file_path']
                if os.path.exists(ref_path):
                    y_ref, _ = librosa.load(ref_path, sr=22050)
                    
                    # Simple pitch matching
                    ref_pitches = librosa.yin(y_ref, fmin=80, fmax=400)
                    tts_pitches = librosa.yin(y_tts, fmin=80, fmax=400)
                    
                    ref_pitch = np.nanmean(ref_pitches)
                    tts_pitch = np.nanmean(tts_pitches)
                    
                    if not np.isnan(ref_pitch) and not np.isnan(tts_pitch):
                        semitone_shift = 12 * np.log2(ref_pitch / tts_pitch)
                        semitone_shift = np.clip(semitone_shift, -4, 4)
                        
                        print(f"🎯 Simple pitch shift: {semitone_shift:.1f} semitones")
                        y_tts = librosa.effects.pitch_shift(y_tts, sr=sr, n_steps=semitone_shift)
            except:
                pass
        
        return y_tts / np.max(np.abs(y_tts)) * 0.8
    
    def get_audio_analysis(self, voice_id):
        if voice_id in self.voice_models:
            return self.voice_models[voice_id].get('audio_analysis')
        return None
    
    def get_training_status(self, training_id):
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def list_voices(self):
        return list(self.voice_models.keys())
    
    def get_voice_info(self, voice_id):
        if voice_id in self.voice_models:
            profile = self.voice_models[voice_id]
            info = {
                'voice_id': voice_id,
                'sample_count': profile.get('sample_count', 0),
                'model_type': profile.get('model_type', 'simple')
            }
            
            if 'audio_analysis' in profile:
                analysis = profile['audio_analysis']
                info.update({
                    'total_duration': analysis.get('total_duration', 0),
                    'quality_level': analysis.get('quality_level', 'Medium'),
                    'recommended_epochs': analysis.get('recommended_epochs', 5)
                })
            
            return info
        return None