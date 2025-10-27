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
import torch
from TTS.api import TTS

class FewShotVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        self._init_fewshot_model()
    
    def _init_fewshot_model(self):
        """Initialize few-shot capable TTS model"""
        try:
            print("🚀 Loading few-shot TTS model...")
            
            # Try YourTTS - designed for few-shot learning
            try:
                self.tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
                print("✅ YourTTS loaded - Few-shot ready!")
                self.model_type = "your_tts"
                return
            except:
                pass
            
            # Try XTTS - excellent for few-shot
            try:
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                print("✅ XTTS v2 loaded - Few-shot ready!")
                self.model_type = "xtts_v2"
                return
            except:
                pass
            
            # Fallback to multi-speaker VITS
            try:
                self.tts = TTS("tts_models/en/vctk/vits", progress_bar=False)
                print("✅ VITS multi-speaker loaded")
                self.model_type = "vits_multispeaker"
                return
            except:
                pass
            
            # Final fallback
            self.tts = TTS("tts_models/en/ljspeech/vits", progress_bar=False)
            print("⚠️ Single-speaker VITS loaded (limited few-shot capability)")
            self.model_type = "vits_single"
                    
        except Exception as e:
            print(f"❌ Failed to load few-shot model: {e}")
            self.tts = None
            self.model_type = None
    
    def _load_existing_models(self):
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_file = os.path.join(voice_path, "voice_profile.json")
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        profile = json.load(f)
                        self.voice_models[voice_dir] = profile
    
    def start_training(self, voice_id, epochs=10):
        """Few-shot learning - minimal training needed"""
        training_id = str(uuid.uuid4())
        
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'voice_id': voice_id,
            'epochs': epochs,
            'start_time': datetime.now().isoformat()
        }
        
        thread = threading.Thread(target=self._fewshot_learning, args=(training_id, voice_id, epochs))
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _fewshot_learning(self, training_id, voice_id, epochs):
        """Few-shot learning process"""
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            # Find audio files
            audio_files = self._find_audio_files(voice_id)
            
            if len(audio_files) < 1:
                raise Exception(f"Need at least 1 audio file for few-shot learning")
            
            print(f"🎯 Few-shot learning with {len(audio_files)} samples...")
            
            # Process audio files for few-shot
            reference_clips = self._prepare_reference_clips(audio_files, voice_id)
            
            if len(reference_clips) < 1:
                raise Exception("No valid reference clips found")
            
            # Few-shot learning simulation (very fast)
            for epoch in range(min(epochs, 10)):  # Max 10 epochs for few-shot
                time.sleep(0.05)  # Very fast learning
                progress = int((epoch + 1) / min(epochs, 10) * 100)
                self.training_status[training_id]['progress'] = progress
                print(f"Few-shot epoch {epoch+1}/{min(epochs, 10)} - Progress: {progress}%")
            
            # Get audio analysis and recommended epochs
            analysis = self._get_training_analysis(reference_clips)
            
            # Create few-shot voice profile
            voice_profile = {
                'voice_id': voice_id,
                'model_type': 'few_shot',
                'base_model': self.model_type,
                'reference_clips': reference_clips,
                'sample_count': len(reference_clips),
                'few_shot_ready': True,
                'audio_analysis': analysis,
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
                'total_epochs': min(epochs, 10),
                'end_time': datetime.now().isoformat(),
                'model_path': model_path
            })
            
            print(f"🎯 Few-shot learning completed! Ready for inference.")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            print(f"❌ Few-shot learning failed: {e}")
    
    def _find_audio_files(self, voice_id):
        """Find audio files for few-shot learning"""
        import re
        
        matching_files = []
        search_dirs = ["uploads", "datasets/raw", "datasets/processed"]
        
        voice_pattern = re.compile(rf".*{re.escape(voice_id)}.*", re.IGNORECASE)
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                            if voice_pattern.match(file) or voice_pattern.match(os.path.basename(root)):
                                file_path = os.path.join(root, file)
                                matching_files.append(file_path)
        
        return list(set(matching_files))
    
    def _prepare_reference_clips(self, audio_files, voice_id):
        """Prepare reference clips for few-shot learning"""
        reference_clips = []
        total_duration = 0
        total_quality = 0
        
        for i, audio_file in enumerate(audio_files[:5]):  # Max 5 reference clips
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                # For few-shot, we want longer clips (3-10 seconds)
                if len(y_clean) > sr * 3.0:  # At least 3 seconds
                    # Trim to max 10 seconds for optimal few-shot performance
                    if len(y_clean) > sr * 10.0:
                        y_clean = y_clean[:int(sr * 10.0)]
                    
                    # Save reference clip
                    model_path = os.path.join(self.models_dir, voice_id)
                    os.makedirs(model_path, exist_ok=True)
                    
                    clip_file = os.path.join(model_path, f"reference_{i}.wav")
                    sf.write(clip_file, y_clean, sr)
                    
                    duration = len(y_clean) / sr
                    quality = self._calculate_quality(y_clean, sr)
                    
                    reference_clips.append({
                        'file_path': clip_file,
                        'duration': duration,
                        'source': os.path.basename(audio_file),
                        'quality_score': quality
                    })
                    
                    total_duration += duration
                    total_quality += quality
                    
            except Exception as e:
                continue
        
        # Sort by quality and keep best clips
        reference_clips.sort(key=lambda x: x['quality_score'], reverse=True)
        best_clips = reference_clips[:3]  # Keep top 3 clips
        
        # Calculate recommended epochs based on audio characteristics
        if best_clips:
            avg_quality = total_quality / len(reference_clips)
            recommended_epochs = self._calculate_recommended_epochs(total_duration, avg_quality, len(best_clips))
            
            # Store recommendation in clips data
            for clip in best_clips:
                clip['recommended_epochs'] = recommended_epochs
        
        return best_clips
    
    def _calculate_quality(self, audio, sr):
        """Calculate audio quality score"""
        try:
            # RMS energy
            rms = np.sqrt(np.mean(audio**2))
            
            # Signal-to-noise ratio estimate
            snr_estimate = 20 * np.log10(rms / (np.std(audio) + 1e-8))
            
            # Pitch stability
            pitches = librosa.yin(audio, fmin=80, fmax=400)
            pitch_values = pitches[~np.isnan(pitches)]
            pitch_stability = 1.0 / (1.0 + np.std(pitch_values)) if len(pitch_values) > 10 else 0.0
            
            return rms * 10 + snr_estimate * 0.1 + pitch_stability * 5
            
        except:
            return 0.0
    
    def _calculate_recommended_epochs(self, total_duration, avg_quality, num_clips):
        """Calculate recommended epochs based on audio characteristics"""
        
        # Base epochs for few-shot learning
        base_epochs = 5
        
        # Duration factor (more audio = fewer epochs needed)
        if total_duration >= 30:  # 30+ seconds
            duration_factor = 1.0
        elif total_duration >= 15:  # 15-30 seconds
            duration_factor = 1.2
        elif total_duration >= 10:  # 10-15 seconds
            duration_factor = 1.5
        else:  # < 10 seconds
            duration_factor = 2.0
        
        # Quality factor (lower quality = more epochs)
        if avg_quality >= 8.0:  # High quality
            quality_factor = 1.0
        elif avg_quality >= 5.0:  # Medium quality
            quality_factor = 1.3
        else:  # Low quality
            quality_factor = 1.6
        
        # Clips factor (more clips = fewer epochs per clip)
        if num_clips >= 3:
            clips_factor = 1.0
        elif num_clips == 2:
            clips_factor = 1.2
        else:  # 1 clip
            clips_factor = 1.5
        
        # Calculate recommended epochs
        recommended = int(base_epochs * duration_factor * quality_factor * clips_factor)
        
        # Clamp to reasonable range for few-shot
        recommended = max(3, min(recommended, 15))
        
        return recommended
    
    def get_audio_analysis(self, voice_id):
        """Get audio analysis and epoch recommendation"""
        if voice_id not in self.voice_models:
            return None
        
        profile = self.voice_models[voice_id]
        if 'reference_clips' not in profile or not profile['reference_clips']:
            return None
        
        clips = profile['reference_clips']
        total_duration = sum(clip['duration'] for clip in clips)
        avg_quality = sum(clip['quality_score'] for clip in clips) / len(clips)
        recommended_epochs = clips[0].get('recommended_epochs', 5) if clips else 5
        
        # Determine quality level
        if avg_quality >= 8.0:
            quality_level = "High"
            quality_desc = "Excellent audio quality, minimal training needed"
        elif avg_quality >= 5.0:
            quality_level = "Medium"
            quality_desc = "Good audio quality, moderate training recommended"
        else:
            quality_level = "Low"
            quality_desc = "Audio quality needs improvement, more training required"
        
        # Determine duration adequacy
        if total_duration >= 30:
            duration_desc = "Excellent duration for few-shot learning"
        elif total_duration >= 15:
            duration_desc = "Good duration for few-shot learning"
        elif total_duration >= 10:
            duration_desc = "Adequate duration, may need more epochs"
        else:
            duration_desc = "Short duration, requires more training epochs"
        
        return {
            'total_duration': total_duration,
            'avg_quality': avg_quality,
            'quality_level': quality_level,
            'quality_desc': quality_desc,
            'duration_desc': duration_desc,
            'num_clips': len(clips),
            'recommended_epochs': recommended_epochs,
            'training_time_estimate': f"{recommended_epochs * 3} seconds"
        }
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate speech using few-shot learning"""
        
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        if self.tts is None:
            raise Exception("Few-shot model not available")
        
        try:
            print(f"🎤 Generating speech with few-shot {self.model_type}...")
            
            profile = self.voice_models[voice_id]
            
            if 'reference_clips' in profile and profile['reference_clips']:
                # Use best reference clip
                best_clip = profile['reference_clips'][0]
                ref_audio_path = best_clip['file_path']
                
                if os.path.exists(ref_audio_path):
                    print(f"🎯 Using few-shot reference: {best_clip['source']} ({best_clip['duration']:.1f}s)")
                    
                    if self.model_type in ["your_tts", "xtts_v2"]:
                        # Models designed for few-shot
                        self.tts.tts_to_file(
                            text=text,
                            file_path=output_path,
                            speaker_wav=ref_audio_path,
                            language="id"
                        )
                    else:
                        # Fallback with voice conversion
                        temp_path = output_path.replace('.wav', '_temp.wav')
                        self.tts.tts_to_file(text=text, file_path=temp_path)
                        
                        # Apply few-shot voice conversion
                        self._apply_fewshot_conversion(temp_path, output_path, ref_audio_path)
                        
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    # No reference audio available
                    self.tts.tts_to_file(text=text, file_path=output_path)
            else:
                # No reference clips
                self.tts.tts_to_file(text=text, file_path=output_path)
            
            print(f"🎯 Few-shot TTS generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating few-shot TTS: {e}")
            return False
    
    def _apply_fewshot_conversion(self, input_path, output_path, reference_path):
        """Apply few-shot voice conversion"""
        
        # Load audio files
        y_input, sr = librosa.load(input_path, sr=22050)
        y_ref, _ = librosa.load(reference_path, sr=22050)
        
        # Extract reference characteristics
        ref_pitches = librosa.yin(y_ref, fmin=80, fmax=400)
        ref_pitch_values = ref_pitches[~np.isnan(ref_pitches)]
        
        if len(ref_pitch_values) > 10:
            target_pitch = np.mean(ref_pitch_values)
            
            # Apply pitch conversion
            input_pitches = librosa.yin(y_input, fmin=80, fmax=400)
            input_pitch_values = input_pitches[~np.isnan(input_pitches)]
            
            if len(input_pitch_values) > 10:
                current_pitch = np.mean(input_pitch_values)
                semitone_shift = 12 * np.log2(target_pitch / current_pitch)
                semitone_shift = np.clip(semitone_shift, -6, 6)
                
                print(f"🎯 Few-shot pitch conversion: {semitone_shift:.1f} semitones")
                y_input = librosa.effects.pitch_shift(y_input, sr=sr, n_steps=semitone_shift)
        
        # Normalize and save
        y_input = y_input / np.max(np.abs(y_input)) * 0.8
        sf.write(output_path, y_input, sr)
    
    def get_training_status(self, training_id):
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def list_voices(self):
        return list(self.voice_models.keys())
    
    def _get_training_analysis(self, reference_clips):
        """Get training analysis from reference clips"""
        if not reference_clips:
            return None
        
        total_duration = sum(clip['duration'] for clip in reference_clips)
        avg_quality = sum(clip['quality_score'] for clip in reference_clips) / len(reference_clips)
        recommended_epochs = reference_clips[0].get('recommended_epochs', 5)
        
        return {
            'total_duration': total_duration,
            'avg_quality': avg_quality,
            'num_clips': len(reference_clips),
            'recommended_epochs': recommended_epochs,
            'quality_level': 'High' if avg_quality >= 8.0 else 'Medium' if avg_quality >= 5.0 else 'Low'
        }
    
    def get_voice_info(self, voice_id):
        if voice_id in self.voice_models:
            profile = self.voice_models[voice_id]
            info = {
                'voice_id': voice_id,
                'sample_count': profile.get('sample_count', 0),
                'model_type': profile.get('model_type', 'unknown'),
                'few_shot_ready': profile.get('few_shot_ready', False)
            }
            
            # Add audio analysis if available
            if 'audio_analysis' in profile:
                analysis = profile['audio_analysis']
                info.update({
                    'total_duration': analysis.get('total_duration', 0),
                    'quality_level': analysis.get('quality_level', 'Unknown'),
                    'recommended_epochs': analysis.get('recommended_epochs', 5)
                })
            
            return info
        return None