from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
from activity_logger import ActivityLogger
from user_manager import UserManager
import os
import json
from datetime import datetime
try:
    from multispeaker_voice_cloner import MultiSpeakerVoiceCloner
    MULTISPEAKER_AVAILABLE = True
except ImportError:
    try:
        from advanced_voice_cloner import AdvancedVoiceCloner as MultiSpeakerVoiceCloner
        MULTISPEAKER_AVAILABLE = False
        print("🚀 Using advanced voice cloner (premium quality)")
    except ImportError:
        try:
            from improved_voice_cloner import ImprovedVoiceCloner as MultiSpeakerVoiceCloner
            MULTISPEAKER_AVAILABLE = False
            print("✨ Using improved voice cloner (advanced fallback)")
        except ImportError:
            from simple_tts_fallback import SimpleTTSFallback as MultiSpeakerVoiceCloner
            MULTISPEAKER_AVAILABLE = False
            print("⚠️ Using basic fallback TTS")
from audio_processor import AudioProcessor

app = Flask(__name__)
app.secret_key = 'ai_dubbing_secret_key_2024'
CORS(app)

# Initialize user manager and activity logger
user_manager = UserManager()
activity_logger = ActivityLogger()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Konfigurasi folder
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
OUTPUT_FOLDER = 'outputs'

for folder in [UPLOAD_FOLDER, MODELS_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Inisialisasi voice cloner
voice_cloner = MultiSpeakerVoiceCloner()
audio_processor = AudioProcessor()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action', 'login')
        
        if action == 'register':
            # Handle registration
            username = request.form['username']
            password = request.form['password']
            full_name = request.form['full_name']
            email = request.form['email']
            
            success, message = user_manager.register_user(username, password, full_name, email)
            
            if success:
                # Log registration
                activity_logger.log_activity(
                    username=username,
                    activity='register',
                    request=request,
                    details={'status': 'success', 'full_name': full_name}
                )
                
                return render_template('login.html', success=message)
            else:
                # Log failed registration
                activity_logger.log_activity(
                    username=username or 'unknown',
                    activity='register_failed',
                    request=request,
                    details={'status': 'failed', 'reason': message}
                )
                
                return render_template('login.html', error=message)
        
        else:
            # Handle login
            username = request.form['username']
            password = request.form['password']
            
            success, user, message = user_manager.authenticate_user(username, password)
            
            if success:
                session['logged_in'] = True
                session['username'] = username
                session['is_admin'] = user.get('is_admin', False)
                session['full_name'] = user.get('full_name', username)
                
                # Log successful login
                activity_logger.log_activity(
                    username=username,
                    activity='login',
                    request=request,
                    details={'status': 'success', 'user_type': 'admin' if user.get('is_admin') else 'user'}
                )
                
                return redirect(url_for('index'))
            else:
                # Log failed login attempt
                activity_logger.log_activity(
                    username=username or 'unknown',
                    activity='login_failed',
                    request=request,
                    details={'status': 'failed', 'reason': 'invalid_credentials'}
                )
                
                return render_template('login.html', error=message)
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'unknown')
    
    # Log logout
    activity_logger.log_activity(
        username=username,
        activity='logout',
        request=request,
        details={'status': 'success'}
    )
    
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/admin')
@login_required
def admin_dashboard():
    # Check if user is admin
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    # Get activity logs and stats
    logs = activity_logger.get_logs(limit=100)
    stats = activity_logger.get_stats()
    
    # Get unique users for filter
    unique_users = list(set(log.get('username') for log in logs if log.get('username')))
    
    return render_template('admin_dashboard.html', 
                         logs=logs, 
                         stats=stats, 
                         unique_users=unique_users)

@app.route('/admin/users')
@login_required
def user_management():
    # Check if user is admin
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    users = user_manager.get_all_users()
    return render_template('admin_users.html', users=users)

@app.route('/api/admin/users', methods=['GET'])
@login_required
def api_get_users():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    
    users = user_manager.get_all_users()
    return jsonify({'users': users})

@app.route('/api/admin/users', methods=['POST'])
@login_required
def api_create_user():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    full_name = data.get('full_name')
    email = data.get('email')
    is_admin = data.get('is_admin', False)
    
    success, message = user_manager.register_user(username, password, full_name, email, is_admin)
    
    if success:
        activity_logger.log_activity(
            username=session.get('username'),
            activity='admin_create_user',
            request=request,
            details={'created_user': username, 'is_admin': is_admin}
        )
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 400

@app.route('/api/admin/users/<username>', methods=['PUT'])
@login_required
def api_update_user(username):
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    
    success, message = user_manager.update_user(
        username,
        full_name=data.get('full_name'),
        email=data.get('email'),
        password=data.get('password'),
        is_admin=data.get('is_admin')
    )
    
    if success:
        activity_logger.log_activity(
            username=session.get('username'),
            activity='admin_update_user',
            request=request,
            details={'updated_user': username, 'changes': data}
        )
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 400

@app.route('/admin/view-passwords')
@login_required
def view_passwords():
    # Check if user is admin
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    users_with_passwords = user_manager.get_all_users_with_passwords()
    
    # Log admin viewing passwords
    activity_logger.log_activity(
        username=session.get('username', 'unknown'),
        activity='view_passwords',
        request=request,
        details={'action': 'admin_viewed_all_passwords', 'users_count': len(users_with_passwords)}
    )
    
    return render_template('view_passwords.html', 
                         users=users_with_passwords,
                         current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/admin/users/<username>', methods=['DELETE'])
@login_required
def delete_user(username):
    # Check if user is admin
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403
    
    success, message = user_manager.delete_user(username)
    
    if success:
        # Log user deletion
        activity_logger.log_activity(
            username=session.get('username', 'unknown'),
            activity='delete_user',
            request=request,
            details={'deleted_user': username, 'status': 'success'}
        )
        
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 400

@app.route('/admin/change-password', methods=['GET', 'POST'])
@login_required
def admin_change_password():
    # Check if user is admin
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        admin_password = request.form['admin_password']
        target_username = request.form['target_username']
        new_password = request.form['new_user_password']
        confirm_password = request.form['confirm_user_password']
        
        # Validate admin password
        admin_username = session.get('username')
        admin_success, admin_user, admin_message = user_manager.authenticate_user(admin_username, admin_password)
        
        if not admin_success:
            users = user_manager.get_all_users()
            return render_template('admin_change_password.html', users=users, error='Invalid admin password')
        
        # Validate passwords match
        if new_password != confirm_password:
            users = user_manager.get_all_users()
            return render_template('admin_change_password.html', users=users, error='New passwords do not match')
        
        # Validate target user exists
        if not user_manager.get_user(target_username):
            users = user_manager.get_all_users()
            return render_template('admin_change_password.html', users=users, error='Target user not found')
        
        # Change password (admin override)
        success, message = user_manager.update_user(target_username, password=new_password)
        
        if success:
            # Log admin password change
            activity_logger.log_activity(
                username=admin_username,
                activity='admin_change_password',
                request=request,
                details={
                    'target_user': target_username,
                    'status': 'success',
                    'admin_action': True
                }
            )
            
            users = user_manager.get_all_users()
            return render_template('admin_change_password.html', users=users, 
                                 success=f'Password changed successfully for user: {target_username}')
        else:
            # Log failed admin password change
            activity_logger.log_activity(
                username=admin_username,
                activity='admin_change_password_failed',
                request=request,
                details={
                    'target_user': target_username,
                    'status': 'failed',
                    'reason': message,
                    'admin_action': True
                }
            )
            
            users = user_manager.get_all_users()
            return render_template('admin_change_password.html', users=users, error=message)
    
    users = user_manager.get_all_users()
    return render_template('admin_change_password.html', users=users)

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        # Validate passwords match
        if new_password != confirm_password:
            return render_template('change_password.html', error='New passwords do not match')
        
        # Change password
        username = session.get('username')
        success, message = user_manager.change_password(username, current_password, new_password)
        
        if success:
            # Log password change
            activity_logger.log_activity(
                username=username,
                activity='change_password',
                request=request,
                details={'status': 'success'}
            )
            
            return render_template('change_password.html', success=message)
        else:
            # Log failed password change
            activity_logger.log_activity(
                username=username,
                activity='change_password_failed',
                request=request,
                details={'status': 'failed', 'reason': message}
            )
            
            return render_template('change_password.html', error=message)
    
    return render_template('change_password.html')

@app.route('/api/upload-voice', methods=['POST'])
@login_required
def upload_voice():
    """Upload file audio untuk voice cloning"""
    try:
        voice_name = request.form.get('voice_name', 'default')
        
        # Handle single file or multiple files
        single_file = request.files.get('audio')
        multiple_files = request.files.getlist('audioFiles')
        
        files_to_process = []
        
        if single_file and single_file.filename != '':
            files_to_process.append(single_file)
        elif multiple_files:
            files_to_process.extend([f for f in multiple_files if f.filename != ''])
        
        if not files_to_process:
            return jsonify({'error': 'No audio files provided'}), 400
        
        # Validasi dan simpan semua file
        allowed_extensions = {'.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.wma'}
        saved_files = []
        
        for i, file in enumerate(files_to_process):
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                continue  # Skip unsupported files
            
            # Simpan file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{voice_name}_{timestamp}_{i:03d}{file_ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            saved_files.append(filepath)
        
        if not saved_files:
            return jsonify({'error': 'No valid audio files to process'}), 400
        
        # Log upload activity
        activity_logger.log_activity(
            username=session.get('username', 'unknown'),
            activity='upload',
            request=request,
            details={
                'voice_name': voice_name,
                'files_count': len(saved_files),
                'total_size': sum(os.path.getsize(f) for f in saved_files if os.path.exists(f))
            }
        )
        
        # Proses audio untuk training (sistem akan otomatis mencari semua file dengan nama voice_name)
        processed_data = audio_processor.process_for_training(saved_files[0], voice_name)  # Pass first file, system will find all
        
        # Get audio analysis and epoch recommendation
        analysis = voice_cloner.get_audio_analysis(voice_name)
        
        response_data = {
            'message': f'{len(saved_files)} audio files uploaded successfully',
            'voice_id': processed_data['voice_id'],
            'samples_count': processed_data['samples_count'],
            'files_uploaded': len(saved_files)
        }
        
        # Add analysis if available
        if analysis:
            response_data['audio_analysis'] = analysis
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-voice', methods=['POST'])
@login_required
def train_voice():
    """Mulai training voice model"""
    try:
        data = request.get_json()
        voice_id = data.get('voice_id')
        epochs = data.get('epochs', 100)
        
        if not voice_id:
            return jsonify({'error': 'Voice ID required'}), 400
        
        # Log training activity
        activity_logger.log_activity(
            username=session.get('username', 'unknown'),
            activity='train',
            request=request,
            details={
                'voice_id': voice_id,
                'epochs': epochs,
                'estimated_time': f"{epochs * 2} minutes"
            }
        )
        
        # Mulai training (async)
        training_id = voice_cloner.start_training(voice_id, epochs)
        
        return jsonify({
            'message': 'Training started',
            'training_id': training_id,
            'estimated_time': f"{epochs * 2} minutes"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-status/<training_id>')
def training_status(training_id):
    """Cek status training"""
    try:
        status = voice_cloner.get_training_status(training_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-speech', methods=['POST'])
@login_required
def generate_speech():
    """Generate speech dari text menggunakan voice yang sudah di-train"""
    try:
        data = request.get_json()
        text = data.get('text')
        voice_id = data.get('voice_id')
        
        if not text or not voice_id:
            return jsonify({'error': 'Text and voice_id required'}), 400
        
        # Generate audio
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{voice_id}_{timestamp}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        success = voice_cloner.generate_speech(voice_id, text, output_path)
        
        # Log generation activity
        activity_logger.log_activity(
            username=session.get('username', 'unknown'),
            activity='generate',
            request=request,
            details={
                'voice_id': voice_id,
                'text_length': len(text),
                'output_file': output_filename,
                'success': success
            }
        )
        
        if success:
            return jsonify({
                'message': 'Speech generated successfully',
                'audio_url': f'/api/download/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated audio file"""
    try:
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voices')
def list_voices():
    """List semua voice yang tersedia"""
    try:
        voices = voice_cloner.list_voices()
        voice_list = []
        
        for voice_id in voices:
            voice_info = voice_cloner.get_voice_info(voice_id)
            if voice_info:
                voice_data = {
                    'id': voice_id,
                    'name': voice_id.replace('_', ' ').title(),
                    'status': 'ready',
                    'sample_count': voice_info.get('sample_count', 0),
                    'model_type': voice_info.get('model_type', 'unknown')
                }
                
                # Add analysis data if available
                if 'total_duration' in voice_info:
                    voice_data.update({
                        'total_duration': voice_info['total_duration'],
                        'quality_level': voice_info.get('quality_level', 'Unknown'),
                        'recommended_epochs': voice_info.get('recommended_epochs', 5)
                    })
                
                voice_list.append(voice_data)
        
        return jsonify({'voices': voice_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-voice/<voice_id>', methods=['DELETE'])
@login_required
def delete_voice(voice_id):
    """Delete a specific voice model (admin only)"""
    # Check if user is admin
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized - Admin access required'}), 403
    
    try:
        import shutil
        
        # Remove from voice_cloner memory
        if voice_id in voice_cloner.voice_models:
            del voice_cloner.voice_models[voice_id]
        
        # Remove model directory
        model_path = os.path.join(MODELS_FOLDER, voice_id)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        
        # Log admin deletion
        activity_logger.log_activity(
            username=session.get('username', 'unknown'),
            activity='delete_voice',
            request=request,
            details={'voice_id': voice_id, 'admin_action': True}
        )
        
        return jsonify({'message': f'Voice "{voice_id}" deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-all-voices', methods=['DELETE'])
@login_required
def delete_all_voices():
    """Delete all voice models (admin only)"""
    # Check if user is admin
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized - Admin access required'}), 403
    
    try:
        import shutil
        
        # Clear voice_cloner memory
        voice_cloner.voice_models.clear()
        
        # Remove all model directories
        if os.path.exists(MODELS_FOLDER):
            for voice_dir in os.listdir(MODELS_FOLDER):
                voice_path = os.path.join(MODELS_FOLDER, voice_dir)
                if os.path.isdir(voice_path):
                    shutil.rmtree(voice_path)
        
        # Log admin deletion
        activity_logger.log_activity(
            username=session.get('username', 'unknown'),
            activity='delete_all_voices',
            request=request,
            details={'admin_action': True}
        )
        
        return jsonify({'message': 'All voices deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)