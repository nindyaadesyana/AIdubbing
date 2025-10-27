import os
import json
import uuid
import threading
import numpy as np
import torch
from datetime import datetime
from TTS.api import TTS
from TTS.tts.utils.speakers import SpeakerManager
import librosa
import soundfile as sf

class MultiSpeakerVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        self.tts_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        
        # Initialize pretrained multi-speaker model
        self._init_pretrained_model()
        
        print("✅ Multi-Speaker Voice Cloner initialized")
    
    def _init_pretrained_model(self):
        """Initialize pretrained YourTTS or VITS multi-speaker model"""
        try:
            print("🔄 Loading pretrained YourTTS model...")
            self.pretrained_tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
            print("✅ YourTTS model loaded successfully")
        except Exception as e:
            try:
                print("🔄 Trying VITS multi-speaker model...")
                self.pretrained_tts = TTS(model_name="tts_models/en/vctk/vits")
                print("✅ VITS multi-speaker model loaded")
            except Exception as e2:
                print(f"⚠️ Could not load pretrained models: {e}, {e2}")
                self.pretrained_tts = None
    
    def _load_existing_models(self):
        """Load existing trained models"""
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                # Check for TTS model files
                model_file = os.path.join(voice_path, "best_model.pth")
                config_file = os.path.join(voice_path, "config.json")
                speakers_file = os.path.join(voice_path, "speakers.json")
                
                if all(os.path.exists(f) for f in [model_file, config_file]):
                    try:
                        # Load custom trained model
                        tts = TTS(model_path=model_file, config_path=config_file)
                        self.tts_models[voice_dir] = tts
                        
                        # Load voice profile
                        profile_file = os.path.join(voice_path, "voice_profile.json")
                        if os.path.exists(profile_file):
                            with open(profile_file, 'r') as f:
                                self.voice_models[voice_dir] = json.load(f)
                        
                        print(f"✅ Loaded trained model: {voice_dir}")
                    except Exception as e:
                        print(f"❌ Error loading model {voice_dir}: {e}")
    
    def start_training(self, voice_id, epochs=200):
        """Start training with multi-speaker VITS"""
        training_id = str(uuid.uuid4())
        
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'voice_id': voice_id,
            'epochs': epochs,
            'start_time': datetime.now().isoformat(),
            'model_type': 'multispeaker_vits'
        }
        
        thread = threading.Thread(
            target=self._train_multispeaker_model,
            args=(training_id, voice_id, epochs)
        )
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _train_multispeaker_model(self, training_id, voice_id, epochs):
        """Train multi-speaker VITS model"""
        try:
            import subprocess
            import sys
            
            self.training_status[training_id]['status'] = 'preparing'
            
            # Prepare dataset
            dataset_path = f"datasets/processed/{voice_id}"
            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset not found: {dataset_path}")
            
            # Create speakers.json
            speakers_file = os.path.join(dataset_path, "speakers.json")
            speakers_data = {voice_id: 0}
            
            with open(speakers_file, 'w') as f:
                json.dump(speakers_data, f, indent=2)
            
            self.training_status[training_id]['status'] = 'training'
            
            # Create training script
            train_script = f"""
import os
import sys
sys.path.append('{os.getcwd()}')

from multispeaker_train import *

# Update progress callback
def update_progress(step, total_steps):
    progress = int((step / total_steps) * 100)
    with open('training_progress_{training_id}.txt', 'w') as f:
        f.write(str(progress))

# Override trainer to report progress
original_train_step = trainer.train_step
def train_step_with_progress(*args, **kwargs):
    result = original_train_step(*args, **kwargs)
    current_step = trainer.total_steps_done
    total_steps = len(trainer.train_loader) * {epochs}
    update_progress(current_step, total_steps)
    return result

trainer.train_step = train_step_with_progress
trainer.fit()
"""
            
            # Write and execute training script
            script_path = f"temp_train_{training_id}.py"
            with open(script_path, 'w') as f:
                f.write(train_script)
            
            # Run training
            process = subprocess.Popen([
                sys.executable, script_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Monitor progress
            progress_file = f'training_progress_{training_id}.txt'
            while process.poll() is None:
                if os.path.exists(progress_file):
                    try:
                        with open(progress_file, 'r') as f:
                            progress = int(f.read().strip())
                            self.training_status[training_id]['progress'] = progress
                    except:
                        pass
                
                import time
                time.sleep(10)
            
            # Check if training completed successfully
            if process.returncode == 0:
                # Move trained model to models directory
                model_output_path = os.path.join(self.models_dir, voice_id)
                os.makedirs(model_output_path, exist_ok=True)
                
                # Copy model files
                import shutil
                for file in ['best_model.pth', 'config.json', 'speakers.json']:
                    src = os.path.join(output_path, file)
                    dst = os.path.join(model_output_path, file)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                
                # Create voice profile
                voice_profile = {
                    'voice_id': voice_id,
                    'model_type': 'multispeaker_vits',
                    'training_epochs': epochs,
                    'created_at': datetime.now().isoformat(),
                    'model_path': model_output_path
                }
                
                profile_file = os.path.join(model_output_path, "voice_profile.json")
                with open(profile_file, 'w') as f:
                    json.dump(voice_profile, f, indent=2)
                
                # Load the trained model
                model_file = os.path.join(model_output_path, "best_model.pth")
                config_file = os.path.join(model_output_path, "config.json")
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    self.tts_models[voice_id] = TTS(model_path=model_file, config_path=config_file)
                    self.voice_models[voice_id] = voice_profile
                
                self.training_status[training_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'end_time': datetime.now().isoformat()
                })
                
                print(f"✅ Multi-speaker training completed for {voice_id}")
            else:
                raise Exception(f"Training failed with return code {process.returncode}")
            
            # Cleanup
            for temp_file in [script_path, progress_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            print(f"❌ Training failed: {e}")
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate speech using trained multi-speaker model"""
        try:
            if voice_id in self.tts_models:
                # Use custom trained model
                print(f"🎤 Generating with trained model: {voice_id}")
                tts = self.tts_models[voice_id]
                
                # Generate with speaker embedding
                tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker=voice_id
                )
                
            elif self.pretrained_tts:
                # Use pretrained model with speaker adaptation
                print(f"🎤 Generating with pretrained model + adaptation: {voice_id}")
                
                # Get reference audio for speaker adaptation
                reference_audio = self._get_reference_audio(voice_id)
                
                if reference_audio:
                    # Generate with speaker conditioning
                    self.pretrained_tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=reference_audio,
                        language="id"
                    )
                else:
                    # Fallback to default speaker
                    self.pretrained_tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        language="id"
                    )
            else:
                raise Exception("No TTS model available")
            
            print(f"✅ Speech generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            return False
    
    def _get_reference_audio(self, voice_id):
        """Get reference audio for speaker adaptation"""
        # Check model directory
        model_path = os.path.join(self.models_dir, voice_id)
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.wav'):
                    return os.path.join(model_path, file)
        
        # Check datasets
        dataset_path = f"datasets/processed/{voice_id}"
        if os.path.exists(dataset_path):
            clips_dir = os.path.join(dataset_path, "clips")
            if os.path.exists(clips_dir):
                wav_files = [f for f in os.listdir(clips_dir) if f.endswith('.wav')]
                if wav_files:
                    return os.path.join(clips_dir, wav_files[0])
        
        return None
    
    def get_audio_analysis(self, voice_id):
        """Get audio analysis for voice"""
        if voice_id in self.voice_models:
            return self.voice_models[voice_id].get('audio_analysis')
        
        # Analyze dataset if available
        dataset_path = f"datasets/processed/{voice_id}"
        if os.path.exists(dataset_path):
            clips_dir = os.path.join(dataset_path, "clips")
            if os.path.exists(clips_dir):
                wav_files = [f for f in os.listdir(clips_dir) if f.endswith('.wav')]
                total_duration = 0
                
                for wav_file in wav_files[:10]:  # Sample first 10 files
                    try:
                        audio_path = os.path.join(clips_dir, wav_file)
                        y, sr = librosa.load(audio_path)
                        total_duration += len(y) / sr
                    except:
                        continue
                
                # Estimate total duration
                estimated_total = total_duration * len(wav_files) / min(10, len(wav_files))
                
                # Recommend epochs based on data amount
                if estimated_total >= 300:  # 5+ minutes
                    recommended_epochs = 100
                elif estimated_total >= 120:  # 2+ minutes
                    recommended_epochs = 150
                else:
                    recommended_epochs = 200
                
                return {
                    'total_duration': estimated_total,
                    'sample_count': len(wav_files),
                    'quality_level': 'High' if estimated_total >= 300 else 'Medium',
                    'recommended_epochs': recommended_epochs,
                    'training_time_estimate': f"{recommended_epochs * 3} minutes"
                }
        
        return None
    
    def get_training_status(self, training_id):
        """Get training status"""
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def list_voices(self):
        """List available voices"""
        voices = set()
        
        # Add trained models
        voices.update(self.voice_models.keys())
        
        # Add available datasets
        datasets_dir = "datasets/processed"
        if os.path.exists(datasets_dir):
            for voice_dir in os.listdir(datasets_dir):
                if os.path.isdir(os.path.join(datasets_dir, voice_dir)):
                    voices.add(voice_dir)
        
        return list(voices)
    
    def get_voice_info(self, voice_id):
        """Get voice information"""
        if voice_id in self.voice_models:
            profile = self.voice_models[voice_id]
            return {
                'voice_id': voice_id,
                'model_type': profile.get('model_type', 'multispeaker_vits'),
                'sample_count': profile.get('sample_count', 0),
                'training_epochs': profile.get('training_epochs', 0),
                'created_at': profile.get('created_at')
            }
        
        # Get analysis for available dataset
        analysis = self.get_audio_analysis(voice_id)
        if analysis:
            return {
                'voice_id': voice_id,
                'model_type': 'dataset_available',
                'sample_count': analysis.get('sample_count', 0),
                'total_duration': analysis.get('total_duration', 0),
                'quality_level': analysis.get('quality_level', 'Unknown'),
                'recommended_epochs': analysis.get('recommended_epochs', 200)
            }
        
        return None