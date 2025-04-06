let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let captureButton = document.getElementById('captureButton');
let uploadButton = document.getElementById('uploadButton');
let fileInput = document.getElementById('fileInput');
let uploadForm = document.getElementById('uploadForm');
let capturedImageLeft = document.getElementById('capturedImageLeft');
let capturedImageCenter = document.getElementById('capturedImageCenter');
let capturedImageRight = document.getElementById('capturedImageRight');
const maxImages = 3;
const imageContainer = document.getElementById('image-container');
const clearAllButton = document.getElementById('clearAllButton');
const aboutBtn = document.getElementById('about-btn');
const helpBtn = document.getElementById('help-btn');
const aboutPage = document.getElementById('about-page');
const helpPage = document.getElementById('help-page');
const backToMainButtons = document.querySelectorAll('.back-to-main');
const doneButton = document.querySelector('.done-button');
const loadingOverlay = document.querySelector('.loading-overlay');
let faceDetected = false;
let isProcessing = false;

let leftImageDataURL = '';
let centerImageDataURL = '';
let rightImageDataURL = '';

function disableButtons() {
    if (uploadButton) {
        uploadButton.disabled = true;
        uploadButton.classList.add('disabled');
    }
    if (captureButton) {
        captureButton.disabled = true;
        captureButton.classList.add('disabled');
    }
    console.log('Buttons disabled'); // Debug log
}

function enableButtons() {
    if (uploadButton) {
        uploadButton.disabled = false;
        uploadButton.classList.remove('disabled');
    }
    if (captureButton) {
        captureButton.disabled = false;
        captureButton.classList.remove('disabled');
    }
    console.log('Buttons enabled'); // Debug log
}

function checkImageState() {
    let hasImages = false;
    for (let i = 0; i < maxImages; i++) {
        const imageKey = `uploadedImage_${i}`;
        if (localStorage.getItem(imageKey)) {
            hasImages = true;
            break;
        }
    }
    
    if (hasImages) {
        disableButtons();
    } else {
        enableButtons();
    }
    return hasImages;
}

// Loading overlay functions
function showLoadingOverlay() {
    // Create loading overlay if it doesn't exist
    let loadingOverlay = document.getElementById('loading');
    if (!loadingOverlay) {
        loadingOverlay = document.createElement('div');
        loadingOverlay.id = 'loading';
        loadingOverlay.className = 'loading-overlay';
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        loadingOverlay.appendChild(spinner);
        
        const message = document.createElement('p');
        message.textContent = 'Processing Image...';
        loadingOverlay.appendChild(message);
        
        // Always append to body to ensure it's full screen
        document.body.appendChild(loadingOverlay);
    } else {
        // Make sure it's a direct child of body
        if (loadingOverlay.parentElement !== document.body) {
            document.body.appendChild(loadingOverlay);
        }
    }
    
    // Apply styles to ensure it covers the entire screen
    loadingOverlay.style.display = 'flex';
    loadingOverlay.style.position = 'fixed';
    loadingOverlay.style.top = '0';
    loadingOverlay.style.left = '0';
    loadingOverlay.style.width = '100vw';
    loadingOverlay.style.height = '100vh';
    loadingOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    loadingOverlay.style.zIndex = '9999';
    loadingOverlay.style.justifyContent = 'center';
    loadingOverlay.style.alignItems = 'center';
    loadingOverlay.style.flexDirection = 'column';
    
    // Style the spinner
    const spinner = loadingOverlay.querySelector('.spinner');
    spinner.style.border = '4px solid rgba(255, 255, 255, 0.3)';
    spinner.style.borderTop = '4px solid #fff';
    spinner.style.borderRadius = '50%';
    spinner.style.width = '40px';
    spinner.style.height = '40px';
    spinner.style.animation = 'spin 1s linear infinite';
    spinner.style.marginBottom = '20px';
    
    // Add animation if it doesn't exist
    if (!document.getElementById('loading-animation')) {
        const style = document.createElement('style');
        style.id = 'loading-animation';
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Prevent scrolling while loading
    document.body.style.overflow = 'hidden';
}

function hideLoadingOverlay() {
    const loadingOverlay = document.getElementById('loading');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
    // Restore scrolling
    document.body.style.overflow = '';
}

function opencvIsReady() {
    console.log('OpenCV.js is ready');
    startCamera();
}

function updateClearButtonState() {
    const clearButton = document.getElementById('clearAllButton');
    if (!clearButton) return;

    let hasImages = false;
    for (let i = 0; i < maxImages; i++) {
        const imageKey = `uploadedImage_${i}`;
        if (localStorage.getItem(imageKey)) {
            hasImages = true;
            break;
        }
    }

    if (hasImages) {
        clearButton.classList.remove('disabled');
    } else {
        clearButton.classList.add('disabled');
    }
}

document.addEventListener("DOMContentLoaded", function() {
    // Initialize button states
    checkImageState();
    updateClearButtonState();

    // Function to save an image to localStorage and display it
    window.saveAndDisplayImage = function(imageData, index, captureMode = false) {
        if (index === undefined) {
            index = findNextAvailableIndex();
        }
        
        if (index >= maxImages) {
            console.warn("Maximum number of images reached");
            return;
        }
        
        // Save to localStorage
        const imageKey = `uploadedImage_${index}`;
        localStorage.setItem(imageKey, imageData);
        
        // Save capture mode status
        localStorage.setItem("captureMode", captureMode);
        
        // Display in container
        displayImages(captureMode);
        
        // Disable buttons after saving image
        disableButtons();
        updateClearButtonState();
        console.log('Image saved and buttons disabled'); // Debug log
    };
    
    // Function to clear all images
    window.clearAllImages = function() {
        // Clear localStorage
        for (let i = 0; i < maxImages; i++) {
            const imageKey = `uploadedImage_${i}`;
            localStorage.removeItem(imageKey);
        }
        
        // Clear capture mode
        localStorage.removeItem("captureMode");
        
        // Clear the display
        if (imageContainer) {
            imageContainer.innerHTML = '';
        }
        
        // Reset capturedImages array if it exists
        if (window.capturedImages) {
            window.capturedImages = [];
        }
        
        // Enable buttons
        enableButtons();
        updateClearButtonState();
        console.log('Images cleared and buttons enabled'); // Debug log
    };

    
    // Function to find the next available index
    function findNextAvailableIndex() {
        for (let i = 0; i < maxImages; i++) {
            const imageKey = `uploadedImage_${i}`;
            if (!localStorage.getItem(imageKey)) {
                return i;
            }
        }
        return maxImages; // All slots are taken
    }
    
    // Function to display images based on mode (single or multi)
    function displayImages(captureMode) {
        // Clear container first
        imageContainer.innerHTML = '';
        
        // Get saved images
        const savedImages = [];
        for (let i = 0; i < maxImages; i++) {
            const imageKey = `uploadedImage_${i}`;
            const savedImage = localStorage.getItem(imageKey);
            if (savedImage) {
                savedImages.push({
                    index: i,
                    data: savedImage
                });
            }
        }
        
        // If no images, just return
        if (savedImages.length === 0) {
            return;
        }
        
        // Display based on mode
        if (captureMode || savedImages.length > 1) {
            // Multi-image mode (capture mode or multiple upload)
            displayMultipleImages(savedImages);
        } else {
            // Single image mode (upload mode with one image)
            displaySingleImage(savedImages[0]);
        }
    }
    
    // Display a single large image
    function displaySingleImage(imageInfo) {
        // Create container
        const singleContainer = document.createElement("div");
        singleContainer.className = "single-image-container";
        
        // Add image label
        const label = document.createElement("div");
        label.className = "image-label";
        
        // Create image
        const imgElement = document.createElement("img");
        imgElement.src = imageInfo.data;
        imgElement.alt = `Uploaded Image`;
        imgElement.className = "single-preview-image";
        
        // Add to container
        singleContainer.appendChild(imgElement);
        singleContainer.appendChild(label);
        imageContainer.appendChild(singleContainer);
    }
    
    // Display multiple images in a row
    function displayMultipleImages(images) {
        const multiContainer = document.createElement("div");
        multiContainer.className = "multi-image-row";
        
        // Create placeholders for all positions
        for (let i = 0; i < maxImages; i++) {
            const placeholder = document.createElement("div");
            placeholder.className = "individual-image-container";
            placeholder.id = `image-container-${i}`;
            
            // Try to find image for this position
            const imageMatch = images.find(img => img.index === i);
            
            if (imageMatch) {
                // We have an image for this position
                const imgElement = document.createElement("img");
                imgElement.src = imageMatch.data;
                imgElement.alt = `Image ${i+1}`;
                imgElement.className = "preview-image";
                placeholder.appendChild(imgElement);
            }
            
            multiContainer.appendChild(placeholder);
        }
        
        imageContainer.appendChild(multiContainer);
    }
    
    // Handle file input changes
    if (fileInput) {
        fileInput.addEventListener("change", function() {
            const files = fileInput.files;
            if (files.length > 0) {
                // Clear previous images
                window.clearAllImages();
                
                // Process files
                const captureMode = files.length > 1; // Multiple files = capture mode display
                
                for (let i = 0; i < Math.min(files.length, maxImages); i++) {
                    const file = files[i];
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        window.saveAndDisplayImage(e.target.result, i, captureMode);
                    };
                    
                    reader.readAsDataURL(file);
                }
            }
        });
    }
    
    // For capturing images directly - to be used in your capture code
    window.captureImage = function(imageData, index) {
        window.saveAndDisplayImage(imageData, index, true); // true = capture mode
    };
    
    // Add event listener to clear all button
    if (clearAllButton) {
        clearAllButton.addEventListener("click", function() {
            window.clearAllImages();
        });
    }
    
    // Initialize display
    const captureMode = localStorage.getItem("captureMode") === "true";
    if (typeof displayImages === 'function') {
        displayImages(captureMode);
    }
    
    console.log('Initialization complete'); // Debug log
});

// Add this to check for existing images on page load
document.addEventListener("DOMContentLoaded", function() {
    // Check if there are any existing images
    for (let i = 0; i < maxImages; i++) {
        const imageKey = `uploadedImage_${i}`;
        if (localStorage.getItem(imageKey)) {
            disableButtons();
            break;
        }
    }
});

function captureFace() {
    if (isProcessing) return;

    showLoadingOverlay();
    let context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    let imageDataURL = canvas.toDataURL('image/png');

    if (!leftImageDataURL) {
        leftImageDataURL = imageDataURL;
        capturedImageLeft.value = imageDataURL;
        faceDetected = true;
        console.log('Left image captured');
    } else if (!centerImageDataURL) {
        centerImageDataURL = imageDataURL;
        capturedImageCenter.value = imageDataURL;
        console.log('Center image captured');
    } else if (!rightImageDataURL) {
        rightImageDataURL = imageDataURL;
        capturedImageRight.value = imageDataURL;
        console.log('Right image captured');
        uploadForm.submit();
        return;
    }

    hideLoadingOverlay();
    if (faceDetected) {
        setTimeout(() => {
            captureFace();
        }, 500);
    }
}

captureButton.addEventListener('click', async function () {
    // First, check if face-api.js is already loaded
    if (typeof faceapi === 'undefined') {
        // Show loading overlay instead of creating a new status message
        showLoadingOverlay();
        
        // Load face-api.js script
        await loadScript('https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js');
        
        // Load the required models
        await faceapi.nets.tinyFaceDetector.loadFromUri('/static/models');
        
        hideLoadingOverlay();
    }
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(async function (stream) {
                // Rest of your camera code...
                // Create container for the face detection interface
                const container = document.createElement('div');
                container.style.position = 'fixed';
                container.style.top = '0';
                container.style.left = '0';
                container.style.width = '100%';
                container.style.height = '100%';
                container.style.backgroundColor = 'rgba(0,0,0,0.8)';
                container.style.zIndex = '997';
                container.style.display = 'flex';
                container.style.flexDirection = 'column';
                container.style.alignItems = 'center';
                container.style.justifyContent = 'center';
                document.body.appendChild(container);
                
                // Create video element
                const video = document.createElement('video');
                video.setAttribute('autoplay', '');
                video.setAttribute('muted', '');
                video.setAttribute('playsinline', '');
                video.srcObject = stream;
                video.style.maxWidth = '100%';
                video.style.maxHeight = '70vh';
                container.appendChild(video);
                
                // Rest of your camera setup code...
                // Create canvas for face detection
                const canvas = document.createElement('canvas');
                canvas.style.position = 'absolute';
                canvas.style.top = video.offsetTop + 'px';
                canvas.style.left = video.offsetLeft + 'px';
                canvas.style.zIndex = '998';
                container.appendChild(canvas);
                
                // Create thumbnails container
                const thumbnailsContainer = document.createElement('div');
                thumbnailsContainer.style.display = 'flex';
                thumbnailsContainer.style.gap = '10px';
                thumbnailsContainer.style.marginTop = '20px';
                container.appendChild(thumbnailsContainer);
                
                // Create capture button
                const captureBtn = document.createElement('button');
                captureBtn.innerText = 'Capture Face (0/3)';
                captureBtn.style.marginTop = '20px';
                captureBtn.style.padding = '10px 20px';
                captureBtn.style.backgroundColor = '#6a5acd';
                captureBtn.style.color = 'white';
                captureBtn.style.border = 'none';
                captureBtn.style.borderRadius = '5px';
                captureBtn.style.cursor = 'pointer';
                container.appendChild(captureBtn);
                
                // Create close button
                const closeBtn = document.createElement('button');
                closeBtn.innerText = 'Close Camera';
                closeBtn.style.position = 'absolute';
                closeBtn.style.top = '10px';
                closeBtn.style.right = '10px';
                closeBtn.style.padding = '5px 10px';
                closeBtn.style.backgroundColor = '#ff4d4d';
                closeBtn.style.color = 'white';
                closeBtn.style.border = 'none';
                closeBtn.style.borderRadius = '5px';
                closeBtn.style.cursor = 'pointer';
                closeBtn.style.zIndex = '1000';
                container.appendChild(closeBtn);
                
                closeBtn.addEventListener('click', function() {
                    // Stop the camera stream and clean up
                    stream.getTracks().forEach(track => track.stop());
                    container.remove();
                });
                
                let capturedImages = [];
                let detectionInterval;
                
                // Start face detection once video is playing
                video.addEventListener('play', function() {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    
                    // Adjust canvas position to overlay video
                    function updateCanvasPosition() {
                        const rect = video.getBoundingClientRect();
                        canvas.style.position = 'absolute';
                        canvas.style.top = rect.top + 'px';
                        canvas.style.left = rect.left + 'px';
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        canvas.style.width = rect.width + 'px';
                        canvas.style.height = rect.height + 'px';
                    }
                    
                    updateCanvasPosition();
                    window.addEventListener('resize', updateCanvasPosition);
                    
                    // Run face detection
                    detectionInterval = setInterval(async () => {
                        if (video.paused || video.ended) return;
                        
                        const detections = await faceapi.detectAllFaces(
                            video, 
                            new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 })
                        );
                        
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        
                        // Draw face detection results
                        if (detections.length > 0) {
                            detections.forEach(detection => {
                                const box = detection.box;
                                ctx.strokeStyle = '#00FF00';
                                ctx.lineWidth = 3;
                                ctx.strokeRect(box.x, box.y, box.width, box.height);
                            });
                        }
                    }, 100);
                });
                
                captureBtn.addEventListener('click', async function() {
                    if (capturedImages.length >= 3) {
                        return;
                    }
                    
                    // Detect faces
                    const detections = await faceapi.detectAllFaces(
                        video,
                        new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 })
                    );
                    
                    if (detections.length === 0) {
                        alert('No face detected! Please position your face in the camera view.');
                        return;
                    }
                    
                    // Find the largest face
                    const largestFace = detections.reduce((prev, current) => 
                        (prev.box.width * prev.box.height > current.box.width * current.box.height) ? prev : current
                    );
                    
                    // Create a temporary canvas for the cropped face
                    const tempCanvas = document.createElement('canvas');
                    const tempCtx = tempCanvas.getContext('2d');
                    
                    // Make it square
                    const size = Math.max(largestFace.box.width, largestFace.box.height);
                    tempCanvas.width = size;
                    tempCanvas.height = size;
                    
                    // Draw the face onto the temporary canvas
                    tempCtx.drawImage(
                        video, 
                        largestFace.box.x, largestFace.box.y, largestFace.box.width, largestFace.box.height,
                        0, 0, size, size
                    );
                    
                    // Resize to model input size (260x260)
                    const finalCanvas = document.createElement('canvas');
                    finalCanvas.width = 260;
                    finalCanvas.height = 260;
                    const finalCtx = finalCanvas.getContext('2d');
                    finalCtx.drawImage(tempCanvas, 0, 0, 260, 260);
                    
                    // Get image data URL
                    const imageDataUrl = finalCanvas.toDataURL('image/jpeg', 0.9);
                    capturedImages.push(imageDataUrl);
                    window.saveAndDisplayImage(imageDataUrl);

                    // Create thumbnail
                    const thumbnail = document.createElement('div');
                    thumbnail.style.width = '80px';
                    thumbnail.style.height = '80px';
                    thumbnail.style.border = '2px solid #6a5acd';
                    thumbnail.style.backgroundImage = `url(${imageDataUrl})`;
                    thumbnail.style.backgroundSize = 'cover';
                    thumbnail.style.backgroundPosition = 'center';
                    thumbnailsContainer.appendChild(thumbnail);
                    
                    captureBtn.innerText = `Capture Face (${capturedImages.length}/3)`;
                    
                    if (capturedImages.length === 3) {
                        captureBtn.innerText = 'Submit Images';
                        captureBtn.style.backgroundColor = '#4CAF50';
                        
                        // Fill hidden form fields
                        document.getElementById('capturedImageLeft').value = capturedImages[0];
                        document.getElementById('capturedImageCenter').value = capturedImages[1];
                        document.getElementById('capturedImageRight').value = capturedImages[2];
                        
                        captureBtn.addEventListener('click', function() {
                            // Stop detection and camera
                            clearInterval(detectionInterval);
                            stream.getTracks().forEach(track => track.stop());
                            container.remove();
                            
                            // Show loading overlay
                            showLoadingOverlay();
                            
                            // Submit the form
                            document.getElementById('uploadForm').submit();
                        }, { once: true });
                    }
                });
            })
            .catch(function (error) {
                console.error('Error accessing the camera:', error);
                alert('Error accessing the camera. Please ensure your camera is enabled and that you have granted permissions.');
            });
    } else {
        alert('Camera access is not supported by this browser. Please try a different browser.');
    }
});

// Helper function to load scripts dynamically
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

uploadButton.addEventListener('click', function() {
    // Trigger the hidden file input
    document.getElementById('fileInput').click();
});

fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        // Optional: Add validation for file type and size
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const maxSize = 5 * 1024 * 1024; // 5MB

        if (!allowedTypes.includes(file.type)) {
            alert('Please upload a valid image (JPEG, PNG, JPG)');
            return;
        }

        if (file.size > maxSize) {
            alert('File is too large. Maximum file size is 5MB.');
            return;
        }

        // Show loading overlay
        showLoadingOverlay();
        
        // Submit the form automatically after file selection
        document.getElementById('uploadForm').submit();
    }
});


aboutBtn.addEventListener('click', () => {
    aboutPage.style.display = 'block';
    document.body.style.overflow = 'hidden';
});

helpBtn.addEventListener('click', () => {
    helpPage.style.display = 'block';
    document.body.style.overflow = 'hidden';
});

backToMainButtons.forEach(button => {
    button.addEventListener('click', () => {
        aboutPage.style.display = 'none';
        helpPage.style.display = 'none';
        document.body.style.overflow = '';
    });
});

doneButton.addEventListener('click', () => {
    uploadForm.submit();
});

window.addEventListener('load', () => {
    opencvIsReady();
});
