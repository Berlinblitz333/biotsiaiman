// Elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const captions = document.getElementById('captions');
const roomIdInput = document.getElementById('roomId');
const joinCallBtn = document.getElementById('joinCall');
const endCallBtn = document.getElementById('endCall');
const toggleVoiceBtn = document.getElementById('toggleVoice');
const toggleSignBtn = document.getElementById('toggleSign');

// PeerJS for video calling
let peer;
let call;
let localStream;

// Join Call
joinCallBtn.addEventListener('click', async () => {
    const roomId = roomIdInput.value.trim();
    if (!roomId) {
        alert('Enter a Room ID!');
        return;
    }

    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localVideo.srcObject = localStream;

        peer = new Peer(roomId, { host: 'peerjs.com', port: 443, path: '/' });

        peer.on('open', (id) => {
            console.log('Connected with ID:', id);
            joinCallBtn.disabled = true;
            endCallBtn.disabled = false;
        });

        peer.on('call', (incomingCall) => {
            call = incomingCall;
            call.answer(localStream);
            call.on('stream', (remoteStream) => {
                remoteVideo.srcObject = remoteStream;
            });
        });
    } catch (error) {
        console.error('Error joining call:', error);
        alert('Error accessing camera/microphone.');
    }
});

// End Call
endCallBtn.addEventListener('click', () => {
    if (call) call.close();
    if (peer) peer.destroy();
    if (localStream) localStream.getTracks().forEach(track => track.stop());
    localVideo.srcObject = null;
    remoteVideo.srcObject = null;
    joinCallBtn.disabled = false;
    endCallBtn.disabled = true;
});

// Voice-to-Text
let recognition;
toggleVoiceBtn.addEventListener('click', () => {
    if (!recognition) {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.onresult = (event) => {
            const transcript = event.results[event.results.length - 1][0].transcript;
            captions.textContent += transcript + ' ';
        };
        recognition.onend = () => {
            if (recognition) recognition.start();
        };
        recognition.start();
        toggleVoiceBtn.textContent = 'Stop Voice-to-Text';
    } else {
        recognition.stop();
        recognition = null;
        toggleVoiceBtn.textContent = 'Toggle Voice-to-Text';
    }
});

// AI-Based ASL Detection (TensorFlow.js Model for Gestures + Fallback to Letters)
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`
});
hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

let signCamera;
let signVideo;
let lastDetected = '';
let gestureModel; // TensorFlow.js model for AI prediction

// Load AI Model (Placeholder - Replace with your trained model)
async function loadGestureModel() {
    // For demo, create a simple model. In reality, train with data from Teachable Machine or custom TF.js.
    gestureModel = tf.sequential();
    gestureModel.add(tf.layers.dense({ inputShape: [63], units: 32, activation: 'relu' })); // 21 landmarks * 3 (x,y,z)
    gestureModel.add(tf.layers.dense({ units: 10, activation: 'softmax' })); // 10 classes (gestures)
    gestureModel.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    // Load pre-trained weights if available (e.g., from a JSON file). For now, it's untrained - add training data.
    console.log('AI Model loaded (placeholder - train for real use)');
}

// Predict Gesture with AI
async function predictGesture(landmarks) {
    if (!gestureModel) return null;
    const input = [];
    landmarks.forEach(lm => input.push(lm.x, lm.y, lm.z)); // Flatten landmarks
    const tensor = tf.tensor2d([input], [1, 63]);
    const prediction = gestureModel.predict(tensor);
    const result = await prediction.data();
    const maxIndex = result.indexOf(Math.max(...result));
    tensor.dispose();
    prediction.dispose();

    // Map index to gesture (customize based on training)
    const gestures = ['Hello', 'Hello', 'Hello', 'Hello', 'Hello', 'Hello', 'Hello', 'Hello', 'Hello', 'Hello'];
    return gestures[maxIndex] || null;
}

hands.onResults(async (results) => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        let detected = await predictGesture(landmarks); // AI prediction first
        if (!detected) {
            detected = detectASLLetter(landmarks); // Fallback to letters
        }
        if (detected && detected !== lastDetected) {
            captions.textContent += detected + ' ';
            lastDetected = detected;
        }
    } else {
        lastDetected = '';
    }
});

// ASL Letter Detection (Fallback)
function detectASLLetter(landmarks) {
    const wrist = landmarks[0];
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];
    const indexMCP = landmarks[5];
    const middleMCP = landmarks[9];
    const ringMCP = landmarks[13];
    const pinkyMCP = landmarks[17];

    const distance = (p1, p2) => Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2);
    const isExtended = (tip, mcp) => tip.y < mcp.y;

    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP) && distance(thumbTip, indexTip) < 0.1) return 'A';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP) && thumbTip.x > indexTip.x) return 'B';
    if (distance(thumbTip, indexTip) < 0.1 && isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP)) return 'C';
    if (isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'D';
    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP) && isExtended(thumbTip, wrist)) return 'E';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'F';
    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'G';
    if (distance(thumbTip, indexTip) > 0.15) return 'H';
    if (!isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'I';
    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP)) return 'J';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP) && thumbTip.x < indexTip.x) return 'K';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP)) return 'L';
    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP)) return 'M';
    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP)) return 'N';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'O';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'P';
    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP)) return 'Q';
    if (isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'R';
    if (!isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'S';
    if (isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'T';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'U';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'V';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP)) return 'W';
    if (!isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'X';
    if (isExtended(indexTip, indexMCP) && isExtended(middleTip, middleMCP) && isExtended(ringTip, ringMCP) && isExtended(pinkyTip, pinkyMCP)) return 'Y';
    if (isExtended(indexTip, indexMCP) && !isExtended(middleTip, middleMCP) && !isExtended(ringTip, ringMCP) && !isExtended(pinkyTip, pinkyMCP)) return 'Z';
    return null;
}

toggleSignBtn.addEventListener('click', async () => {
    if (!signCamera) {
        try {
            await loadGestureModel(); // Load AI model
            signCamera = await navigator.mediaDevices.getUserMedia({ video: true });
            signVideo = document.createElement('video');
            signVideo.srcObject = signCamera;
            signVideo.style.display = 'none';
            document.body.appendChild(signVideo);
            signVideo.play();

            setInterval(() => {
                hands.send({ image: signVideo });
            }, 500);
            toggleSignBtn.textContent = 'Stop ASL-to-Text';
        } catch (error) {
            console.error('Error accessing camera for sign:', error);
            alert('Error accessing camera.');
        }
    } else {
        signCamera.getTracks().forEach(track => track.stop());
        signCamera = null;
        if (signVideo) signVideo.remove();
        toggleSignBtn.textContent = 'Toggle ASL-to-Text';
    }
});
