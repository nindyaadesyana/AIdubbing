// Modern AI Voice Cloning App JavaScript
let currentVoiceId = null;
let currentTrainingId = null;

// Tab functionality
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    // Load voices when switching to generate or voices tab
    if (tabName === 'generate' || tabName === 'voices') {
        loadVoices();
    }
}

// Single file upload handling
document.getElementById('audioFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // Clear multiple files selection
        document.getElementById('audioFiles').value = '';
        
        const placeholder = this.parentElement.querySelector('.upload-placeholder');
        placeholder.innerHTML = `
            <i class="fas fa-file-audio"></i>
            <p><strong>${file.name}</strong></p>
            <small>Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</small>
        `;
        
        document.getElementById('filesList').style.display = 'none';
    }
});

// Multiple files upload handling
document.getElementById('audioFiles').addEventListener('change', function(e) {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
        // Clear single file selection
        document.getElementById('audioFile').value = '';
        
        // Reset single file placeholder
        const singlePlaceholder = document.getElementById('audioFile').parentElement.querySelector('.upload-placeholder');
        singlePlaceholder.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Upload single audio file</p>
            <small>Supported: MP3, M4A, WAV, FLAC, AAC, OGG, WMA</small>
        `;
        
        displayFilesList(files);
    }
});

// Display files list
function displayFilesList(files) {
    const filesList = document.getElementById('filesList');
    const audioFiles = files.filter(file => {
        const ext = file.name.toLowerCase().split('.').pop();
        return ['mp3', 'm4a', 'wav', 'flac', 'aac', 'ogg', 'wma'].includes(ext);
    });
    
    if (audioFiles.length === 0) {
        filesList.innerHTML = `
            <div style="text-align: center; color: #e53e3e; padding: 20px;">
                <i class="fas fa-exclamation-triangle"></i>
                <p>No audio files found in selected folder</p>
            </div>
        `;
        filesList.style.display = 'block';
        return;
    }
    
    const totalSize = audioFiles.reduce((sum, file) => sum + file.size, 0);
    
    let html = `
        <div class="files-summary">
            <h4><i class="fas fa-folder-open"></i> ${audioFiles.length} Audio Files Selected</h4>
            <p>Total size: ${(totalSize / 1024 / 1024).toFixed(2)} MB</p>
        </div>
    `;
    
    audioFiles.slice(0, 10).forEach(file => {
        html += `
            <div class="file-item">
                <i class="fas fa-file-audio"></i>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${(file.size / 1024 / 1024).toFixed(2)} MB</div>
                </div>
            </div>
        `;
    });
    
    if (audioFiles.length > 10) {
        html += `
            <div class="file-item" style="text-align: center; color: #718096;">
                <i class="fas fa-ellipsis-h"></i>
                <div class="file-info">
                    <div class="file-name">... and ${audioFiles.length - 10} more files</div>
                </div>
            </div>
        `;
    }
    
    filesList.innerHTML = html;
    filesList.style.display = 'block';
}

// Text statistics
document.getElementById('inputText').addEventListener('input', function(e) {
    const text = e.target.value;
    const charCount = text.length;
    const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
    const estimatedDuration = Math.ceil(wordCount * 0.6); // ~0.6 seconds per word
    
    document.getElementById('charCount').textContent = `${charCount} characters`;
    document.getElementById('wordCount').textContent = `${wordCount} words`;
    document.getElementById('estimatedDuration').textContent = `~${estimatedDuration}s duration`;
});

// Upload form handler
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    const singleFile = document.getElementById('audioFile').files[0];
    const multipleFiles = document.getElementById('audioFiles').files;
    const voiceName = document.getElementById('voiceName').value;
    
    if (!voiceName) {
        showStatus('uploadStatus', 'Please enter voice name', 'error');
        return;
    }
    
    if (!singleFile && multipleFiles.length === 0) {
        showStatus('uploadStatus', 'Please select audio file(s)', 'error');
        return;
    }
    
    // Handle single or multiple files
    const allowedFormats = ['.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.wma'];
    
    if (singleFile) {
        // Single file upload
        const fileExt = singleFile.name.toLowerCase().substring(singleFile.name.lastIndexOf('.'));
        if (!allowedFormats.includes(fileExt)) {
            showStatus('uploadStatus', `Format file tidak didukung. Gunakan: ${allowedFormats.join(', ')}`, 'error');
            return;
        }
        formData.append('audio', singleFile);
    } else {
        // Multiple files upload
        const audioFiles = Array.from(multipleFiles).filter(file => {
            const ext = '.' + file.name.toLowerCase().split('.').pop();
            return allowedFormats.includes(ext);
        });
        
        if (audioFiles.length === 0) {
            showStatus('uploadStatus', 'No valid audio files found in selected folder', 'error');
            return;
        }
        
        audioFiles.forEach(file => {
            formData.append('audioFiles', file);
        });
    }
    
    formData.append('voice_name', voiceName);
    
    const fileCount = singleFile ? 1 : Array.from(multipleFiles).length;
    showStatus('uploadStatus', `
        <i class="fas fa-spinner fa-spin"></i> 
        Processing ${fileCount} audio file(s)... 
        <br><small>Searching for all files containing "${voiceName}" and preparing for multi-file training...</small>
    `, 'info');
    
    try {
        const response = await fetch('/api/upload-voice', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentVoiceId = result.voice_id;
            showStatus('uploadStatus', `
                <i class="fas fa-check-circle"></i> 
                Voice uploaded successfully! 
                <br><small>Found ${result.samples_count} audio samples for few-shot training</small>
            `, 'success');
            
            // Show epoch recommendation if available
            if (result.audio_analysis) {
                showEpochRecommendation(result.audio_analysis);
            }
            
            // Show training card
            document.getElementById('trainingCard').style.display = 'block';
            document.getElementById('trainingCard').scrollIntoView({ behavior: 'smooth' });
        } else {
            showStatus('uploadStatus', `<i class="fas fa-exclamation-triangle"></i> ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus('uploadStatus', `<i class="fas fa-exclamation-triangle"></i> Upload failed: ${error.message}`, 'error');
    }
});

// Start training
document.getElementById('startTraining').addEventListener('click', async () => {
    if (!currentVoiceId) {
        showStatus('uploadStatus', '<i class="fas fa-exclamation-triangle"></i> Please upload a voice first', 'error');
        return;
    }
    
    const epochs = document.getElementById('epochs').value;
    
    try {
        const response = await fetch('/api/train-voice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                voice_id: currentVoiceId,
                epochs: parseInt(epochs)
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentTrainingId = result.training_id;
            document.getElementById('trainingStatus').style.display = 'block';
            document.getElementById('trainingMessage').innerHTML = `
                <i class="fas fa-brain"></i> Multi-file training started! 
                <br><small>Processing multiple audio clips with optimized hyperparameters...</small>
            `;
            
            // Start polling training status
            pollTrainingStatus();
        } else {
            showStatus('uploadStatus', `<i class="fas fa-exclamation-triangle"></i> Training failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus('uploadStatus', `<i class="fas fa-exclamation-triangle"></i> Training failed: ${error.message}`, 'error');
    }
});

// Poll training status
function pollTrainingStatus() {
    if (!currentTrainingId) return;
    
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/api/training-status/${currentTrainingId}`);
            const status = await response.json();
            
            const progress = status.progress || 0;
            const currentLoss = status.current_loss || '-';
            const clipsProcessed = status.clips_processed || '-';
            
            // Update progress bar
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressPercent').textContent = progress + '%';
            document.getElementById('progressEpoch').textContent = `Epoch ${Math.floor(progress * (status.epochs || 100) / 100)}/${status.epochs || 100}`;
            
            // Update metrics
            document.getElementById('currentLoss').textContent = typeof currentLoss === 'number' ? currentLoss.toFixed(4) : currentLoss;
            document.getElementById('clipsCount').textContent = clipsProcessed;
            
            if (status.status === 'completed') {
                document.getElementById('progressFill').style.width = '100%';
                document.getElementById('trainingMessage').innerHTML = `
                    <i class="fas fa-check-circle"></i> Multi-file training completed successfully! 
                    <br><small>Final loss: ${status.final_loss?.toFixed(4) || 'N/A'} | 
                    Clips processed: ${status.clips_processed || 'N/A'} | 
                    Epochs: ${status.total_epochs || 'N/A'}</small>
                `;
                clearInterval(interval);
                loadVoices(); // Refresh voice list
            } else if (status.status === 'failed') {
                document.getElementById('trainingMessage').innerHTML = `
                    <i class="fas fa-exclamation-triangle"></i> Training failed: ${status.error}
                `;
                clearInterval(interval);
            } else if (status.status === 'training') {
                document.getElementById('trainingMessage').innerHTML = `
                    <i class="fas fa-cogs fa-spin"></i> Multi-file training in progress... 
                    <br><small>Processing audio clips with batch size 16, optimized hyperparameters</small>
                `;
            }
        } catch (error) {
            console.error('Error polling training status:', error);
        }
    }, 3000); // Poll every 3 seconds
}

// Generate speech form handler
document.getElementById('generateForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const voiceId = document.getElementById('selectedVoice').value;
    const text = document.getElementById('inputText').value;
    
    if (!voiceId || !text) {
        showStatus('generateStatus', '<i class="fas fa-exclamation-triangle"></i> Please select a voice and enter text', 'error');
        return;
    }
    
    showStatus('generateStatus', `
        <i class="fas fa-magic fa-spin"></i> 
        Generating speech with multi-file trained voice... 
        <br><small>Applying voice characteristics from ${text.split(' ').length} words</small>
    `, 'info');
    
    try {
        const response = await fetch('/api/generate-speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                voice_id: voiceId,
                text: text
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus('generateStatus', `
                <i class="fas fa-check-circle"></i> 
                Speech generated successfully! 
                <br><small>Multi-file voice conversion applied</small>
            `, 'success');
            
            // Show audio result
            const audioResult = document.getElementById('audioResult');
            const resultAudio = document.getElementById('resultAudio');
            const downloadLink = document.getElementById('downloadLink');
            
            resultAudio.src = result.audio_url;
            downloadLink.href = result.audio_url;
            downloadLink.download = `${voiceId}_speech_${Date.now()}.wav`;
            audioResult.style.display = 'block';
            audioResult.scrollIntoView({ behavior: 'smooth' });
        } else {
            showStatus('generateStatus', `<i class="fas fa-exclamation-triangle"></i> ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus('generateStatus', `<i class="fas fa-exclamation-triangle"></i> Generation failed: ${error.message}`, 'error');
    }
});

// Voice selection handler
document.getElementById('selectedVoice').addEventListener('change', function(e) {
    const voiceId = e.target.value;
    const voiceInfo = document.getElementById('voiceInfo');
    
    if (voiceId) {
        // Show voice info (you can fetch detailed info from API)
        voiceInfo.style.display = 'block';
        document.getElementById('voiceClips').textContent = 'Loading...';
        document.getElementById('voiceDuration').textContent = 'Loading...';
        document.getElementById('voiceQuality').textContent = 'Loading...';
        
        // Fetch voice details
        fetchVoiceDetails(voiceId);
    } else {
        voiceInfo.style.display = 'none';
    }
});

// Fetch voice details
async function fetchVoiceDetails(voiceId) {
    try {
        // This would be an API call to get voice details
        // For now, we'll use placeholder data
        document.getElementById('voiceClips').textContent = '100';
        document.getElementById('voiceDuration').textContent = '377';
        document.getElementById('voiceQuality').textContent = 'High';
    } catch (error) {
        console.error('Error fetching voice details:', error);
    }
}

// Load available voices
async function loadVoices() {
    try {
        const response = await fetch('/api/voices');
        const result = await response.json();
        
        // Update select dropdown
        const select = document.getElementById('selectedVoice');
        select.innerHTML = '<option value="">Choose a trained voice model...</option>';
        
        // Update voices grid
        const voicesGrid = document.getElementById('voicesList');
        voicesGrid.innerHTML = '';
        
        if (result.voices && result.voices.length > 0) {
            result.voices.forEach(voice => {
                // Add to select dropdown
                const option = document.createElement('option');
                option.value = voice.id;
                option.textContent = voice.name;
                select.appendChild(option);
                
                // Add to voices grid
                const voiceCard = document.createElement('div');
                voiceCard.className = 'voice-card';
                
                // Check if user is admin (from session)
                const isAdmin = document.body.dataset.isAdmin === 'true';
                
                voiceCard.innerHTML = `
                    ${isAdmin ? `<div class="voice-select">
                        <input type="checkbox" class="voice-checkbox" value="${voice.id}" onchange="updateDeleteButton()">
                    </div>` : ''}
                    <h3><i class="fas fa-user-circle"></i> ${voice.name}</h3>
                    <p><i class="fas fa-tag"></i> ID: ${voice.id}</p>
                    <p><i class="fas fa-file-audio"></i> Clips: ${voice.sample_count || 'N/A'}</p>
                    <p><i class="fas fa-clock"></i> Duration: ${voice.total_duration ? voice.total_duration.toFixed(1) + 's' : 'N/A'}</p>
                    <p><i class="fas fa-chart-bar"></i> Avg Clip: ${voice.avg_clip_duration ? voice.avg_clip_duration.toFixed(1) + 's' : 'N/A'}</p>
                    <span class="voice-status ready">
                        <i class="fas fa-check-circle"></i> Ready
                    </span>
                `;
                voicesGrid.appendChild(voiceCard);
            });
        } else {
            voicesGrid.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #718096;">
                    <i class="fas fa-microphone-slash" style="font-size: 3rem; margin-bottom: 20px; opacity: 0.5;"></i>
                    <h3>No Voice Models Found</h3>
                    <p>Upload and train a voice model first to get started.</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading voices:', error);
        document.getElementById('voicesList').innerHTML = `
            <div style="text-align: center; padding: 40px; color: #e53e3e;">
                <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 20px;"></i>
                <h3>Error Loading Voices</h3>
                <p>Please try refreshing the page.</p>
            </div>
        `;
    }
}

// Show epoch recommendation
function showEpochRecommendation(analysis) {
    const recommendationDiv = document.getElementById('epochRecommendation');
    const epochsInput = document.getElementById('epochs');
    
    if (analysis && analysis.recommended_epochs) {
        // Update input value
        epochsInput.value = analysis.recommended_epochs;
        
        // Update recommendation display
        document.getElementById('recommendedEpochs').textContent = analysis.recommended_epochs;
        document.getElementById('audioDuration').textContent = analysis.total_duration?.toFixed(1) || '-';
        document.getElementById('audioQuality').textContent = analysis.quality_level || '-';
        document.getElementById('trainingTime').textContent = analysis.training_time_estimate || '-';
        
        // Update reason
        let reason = 'Few-shot learning optimized';
        if (analysis.quality_level === 'High' && analysis.total_duration >= 30) {
            reason = 'High quality audio with good duration - minimal training needed';
        } else if (analysis.quality_level === 'Low' || analysis.total_duration < 10) {
            reason = 'Audio needs more training due to quality/duration';
        }
        document.getElementById('recommendationReason').textContent = reason;
        
        // Show recommendation
        recommendationDiv.style.display = 'block';
    }
}

// Show status message
function showStatus(elementId, message, type) {
    const statusElement = document.getElementById(elementId);
    statusElement.innerHTML = message;
    statusElement.className = `modern-status ${type}`;
    statusElement.style.display = 'block';
}

// Play audio function
function playAudio() {
    const audio = document.getElementById('resultAudio');
    if (audio.paused) {
        audio.play();
    } else {
        audio.pause();
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    loadVoices();
    
    // Add smooth scrolling for better UX
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});

// Add loading states for better UX
function addLoadingState(buttonId) {
    const button = document.getElementById(buttonId);
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    button.disabled = true;
    
    return () => {
        button.innerHTML = originalText;
        button.disabled = false;
    };
}

// Enhanced error handling
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showStatus('uploadStatus', '<i class="fas fa-exclamation-triangle"></i> An unexpected error occurred. Please try again.', 'error');
});

// Delete single voice (admin only)
async function deleteVoice(voiceId) {
    if (!confirm(`Delete voice "${voiceId}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/delete-voice/${voiceId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus('uploadStatus', `<i class="fas fa-check-circle"></i> Voice "${voiceId}" deleted`, 'success');
            loadVoices();
        } else {
            showStatus('uploadStatus', `<i class="fas fa-exclamation-triangle"></i> Failed to delete: ${result.error}`, 'error');
        }
    } catch (error) {
        showStatus('uploadStatus', `<i class="fas fa-exclamation-triangle"></i> Error: ${error.message}`, 'error');
    }
}

// Delete selected voices (admin only)
function deleteSelectedVoices() {
    const checkboxes = document.querySelectorAll('.voice-checkbox:checked');
    const selectedVoices = Array.from(checkboxes).map(cb => cb.value);
    
    if (selectedVoices.length === 0) {
        alert('Please select voices to delete');
        return;
    }
    
    if (!confirm(`Delete ${selectedVoices.length} selected voice(s)?`)) {
        return;
    }
    
    selectedVoices.forEach(voiceId => {
        deleteVoice(voiceId);
    });
}

// Toggle all checkboxes (admin only)
function toggleAllVoices() {
    const selectAll = document.getElementById('selectAllVoices');
    const checkboxes = document.querySelectorAll('.voice-checkbox');
    
    checkboxes.forEach(cb => {
        cb.checked = selectAll.checked;
    });
    
    updateDeleteButton();
}

// Update delete button state (admin only)
function updateDeleteButton() {
    const deleteBtn = document.getElementById('deleteSelectedBtn');
    if (!deleteBtn) return; // Button doesn't exist for regular users
    
    const checkboxes = document.querySelectorAll('.voice-checkbox:checked');
    const count = checkboxes.length;
    
    if (count > 0) {
        deleteBtn.textContent = `Delete Selected (${count})`;
        deleteBtn.disabled = false;
    } else {
        deleteBtn.textContent = 'Delete Selected';
        deleteBtn.disabled = true;
    }
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit forms
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const activeTab = document.querySelector('.tab-content.active');
        const form = activeTab.querySelector('form');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
});