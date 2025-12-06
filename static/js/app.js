// Kavach Web Application JavaScript

const API_BASE = '';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const processBtn = document.getElementById('processBtn');
const resultsSection = document.getElementById('resultsSection');
const resultContainer = document.getElementById('resultContainer');
const downloadBtn = document.getElementById('downloadBtn');
const gallery = document.getElementById('gallery');
const galleryLoading = document.getElementById('galleryLoading');
const liveBtn = document.getElementById('liveBtn');
const liveSection = document.getElementById('liveSection');
const webcamVideo = document.getElementById('webcamVideo');
const processedCanvas = document.getElementById('processedCanvas');
const livePlaceholder = document.getElementById('livePlaceholder');
const liveStatus = document.getElementById('liveStatus');
const liveConfSlider = document.getElementById('liveConfSlider');
const liveConfValue = document.getElementById('liveConfValue');
const fpsCounter = document.getElementById('fpsCounter');

let selectedFile = null;
let isLiveMode = false;
let stream = null;
let frameCount = 0;
let lastFpsTime = Date.now();
let fps = 0;
let processingFrame = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadPastRuns();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove file
    removeFileBtn.addEventListener('click', removeFile);
    
    // Confidence slider
    confSlider.addEventListener('input', (e) => {
        confValue.textContent = e.target.value + '%';
    });
    
    // Process button
    processBtn.addEventListener('click', processFile);
    
    // Download button
    downloadBtn.addEventListener('click', downloadResult);
    
    // Live detection button
    liveBtn.addEventListener('click', toggleLiveDetection);
    
    // Live confidence slider
    liveConfSlider.addEventListener('input', (e) => {
        liveConfValue.textContent = e.target.value + '%';
    });
}

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        setSelectedFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        setSelectedFile(file);
    }
}

function setSelectedFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'flex';
    uploadArea.style.display = 'none';
    processBtn.disabled = false;
}

function removeFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    uploadArea.style.display = 'block';
    processBtn.disabled = true;
    resultsSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Processing
async function processFile() {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('conf', confSlider.value / 100);
    
    // Update UI
    processBtn.disabled = true;
    processBtn.querySelector('.btn-text').style.display = 'none';
    processBtn.querySelector('.btn-loader').style.display = 'inline';
    
    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showResult(data.result_file, data.file_type);
            showNotification('Processing completed successfully!', 'success');
            loadPastRuns(); // Refresh gallery
        } else {
            showNotification(data.error || 'Processing failed', 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    } finally {
        processBtn.disabled = false;
        processBtn.querySelector('.btn-text').style.display = 'inline';
        processBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

function showResult(filename, fileType) {
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    const resultUrl = `${API_BASE}/api/results/${filename}`;
    
    if (fileType === 'image') {
        resultContainer.innerHTML = `
            <img src="${resultUrl}" alt="Result" class="result-image">
        `;
    } else {
        resultContainer.innerHTML = `
            <video src="${resultUrl}" controls class="result-video"></video>
        `;
    }
    
    downloadBtn.style.display = 'block';
    downloadBtn.dataset.filename = filename;
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function downloadResult() {
    const filename = downloadBtn.dataset.filename;
    if (filename) {
        window.open(`${API_BASE}/api/results/${filename}`, '_blank');
    }
}

// Past Runs
async function loadPastRuns() {
    galleryLoading.style.display = 'block';
    gallery.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/past-runs`);
        const data = await response.json();
        
        galleryLoading.style.display = 'none';
        
        if (data.results && data.results.length > 0) {
            data.results.forEach(item => {
                const galleryItem = createGalleryItem(item);
                gallery.appendChild(galleryItem);
            });
        } else {
            gallery.innerHTML = '<div class="loading">No past runs found</div>';
        }
    } catch (error) {
        galleryLoading.style.display = 'none';
        gallery.innerHTML = '<div class="loading">Error loading past runs</div>';
    }
}

function createGalleryItem(item) {
    const div = document.createElement('div');
    div.className = 'gallery-item fade-in';
    div.dataset.filename = item.filename;
    
    const resultUrl = `${API_BASE}/api/results/${item.filename}`;
    const date = new Date(item.modified).toLocaleDateString();
    
    if (item.type === 'image') {
        div.innerHTML = `
            <div class="gallery-item-media">
                <img src="${resultUrl}" alt="${item.filename}" loading="lazy">
                <button class="gallery-delete-btn" onclick="event.stopPropagation(); deleteGalleryItem('${item.filename}')" title="Delete">
                    🗑️
                </button>
            </div>
            <div class="gallery-item-info">
                <div class="gallery-item-name">${item.filename}</div>
                <div class="gallery-item-date">${date}</div>
            </div>
        `;
    } else {
        div.innerHTML = `
            <div class="gallery-item-media">
                <video src="${resultUrl}" muted></video>
                <button class="gallery-delete-btn" onclick="event.stopPropagation(); deleteGalleryItem('${item.filename}')" title="Delete">
                    🗑️
                </button>
            </div>
            <div class="gallery-item-info">
                <div class="gallery-item-name">${item.filename}</div>
                <div class="gallery-item-date">${date}</div>
            </div>
        `;
    }
    
    // Click to view full size
    div.addEventListener('click', (e) => {
        if (!e.target.classList.contains('gallery-delete-btn')) {
            window.open(resultUrl, '_blank');
        }
    });
    
    return div;
}

async function deleteGalleryItem(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/results/${filename}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showNotification('File deleted successfully', 'success');
            // Remove the item from the gallery
            const item = document.querySelector(`[data-filename="${filename}"]`);
            if (item) {
                item.style.animation = 'fadeOut 0.3s ease';
                setTimeout(() => {
                    item.remove();
                    // Reload gallery if empty
                    if (gallery.children.length === 0) {
                        loadPastRuns();
                    }
                }, 300);
            }
        } else {
            showNotification(data.error || 'Failed to delete file', 'error');
        }
    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
    }
}

// Live Detection
async function toggleLiveDetection() {
    if (!isLiveMode) {
        await startLiveDetection();
    } else {
        stopLiveDetection();
    }
}

async function startLiveDetection() {
    try {
        // Request webcam access
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        webcamVideo.srcObject = stream;
        isLiveMode = true;
        liveSection.style.display = 'block';
        liveBtn.querySelector('.btn-text').style.display = 'none';
        liveBtn.querySelector('.btn-stop-text').style.display = 'inline';
        liveBtn.classList.add('active');
        
        // Hide confidence and FPS controls when webcam is running
        const liveControls = document.querySelector('.live-controls');
        if (liveControls) {
            liveControls.style.display = 'none';
        }
        
        // Setup canvas
        const ctx = processedCanvas.getContext('2d');
        processedCanvas.width = 1280;
        processedCanvas.height = 720;
        
        livePlaceholder.style.display = 'none';
        processedCanvas.style.display = 'block';
        
        // Start processing frames
        processLiveFrame();
        
        // Scroll to live section
        liveSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } catch (error) {
        showNotification('Error accessing webcam: ' + error.message, 'error');
        console.error('Webcam error:', error);
    }
}

function stopLiveDetection() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    isLiveMode = false;
    liveSection.style.display = 'none';
    liveBtn.querySelector('.btn-text').style.display = 'inline';
    liveBtn.querySelector('.btn-stop-text').style.display = 'none';
    liveBtn.classList.remove('active');
    
    // Show confidence and FPS controls again when stopped
    const liveControls = document.querySelector('.live-controls');
    if (liveControls) {
        liveControls.style.display = 'flex';
    }
    
    webcamVideo.srcObject = null;
    processingFrame = false;
    fps = 0;
    frameCount = 0;
}

async function processLiveFrame() {
    if (!isLiveMode || processingFrame) {
        if (isLiveMode) {
            requestAnimationFrame(() => processLiveFrame());
        }
        return;
    }
    
    processingFrame = true;
    
    try {
        // Capture frame from video
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth || 1280;
        canvas.height = webcamVideo.videoHeight || 720;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0);
        
        // Convert to blob
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            formData.append('conf', liveConfSlider.value / 100);
            
            try {
                const response = await fetch(`${API_BASE}/api/process-frame`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok && data.image) {
                    // Display processed frame
                    const img = new Image();
                    img.onload = () => {
                        const processedCtx = processedCanvas.getContext('2d');
                        processedCanvas.width = img.width;
                        processedCanvas.height = img.height;
                        processedCtx.drawImage(img, 0, 0);
                        
                        // Update FPS
                        frameCount++;
                        const now = Date.now();
                        if (now - lastFpsTime >= 1000) {
                            fps = frameCount;
                            frameCount = 0;
                            lastFpsTime = now;
                            fpsCounter.textContent = fps;
                        }
                    };
                    img.src = data.image;
                    
                    // Show notification if face was detected and saved
                    if (data.has_faces && data.saved) {
                        // Optional: show a brief indicator that photo was saved
                        // (commented out to avoid spam, but can be enabled)
                        // showNotification('📸 Photo saved (face detected)', 'success');
                    }
                }
            } catch (error) {
                console.error('Processing error:', error);
            } finally {
                processingFrame = false;
                if (isLiveMode) {
                    // Process next frame after a delay (to control FPS)
                    setTimeout(() => {
                        requestAnimationFrame(() => processLiveFrame());
                    }, 100); // ~10 FPS processing
                }
            }
        }, 'image/jpeg', 0.8);
        
    } catch (error) {
        console.error('Frame capture error:', error);
        processingFrame = false;
        if (isLiveMode) {
            requestAnimationFrame(() => processLiveFrame());
        }
    }
}

// Notifications
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

