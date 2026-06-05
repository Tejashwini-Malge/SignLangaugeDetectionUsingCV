# HANDSPEAK — Real-Time ASL Detection & Voice Synthesis

A web-based real-time American Sign Language (ASL) detection system powered by AI. **HANDSPEAK** enables deaf and non-verbal users to communicate through sign language, with the system automatically converting signs to natural speech in multiple languages.

**Live hand tracking → LSTM sign recognition → LLM sentence formation → Multi-language text-to-speech**

---

## 🎯 Features

- **Real-Time Hand Tracking**: MediaPipe Hands detects 21 hand landmarks at 10fps
- **Sign Recognition**: LSTM neural network trained on custom ASL gesture sequences
- **Intelligent Segmentation**: Automatic detection of word boundaries and sentence breaks
- **AI-Powered Sentences**: Groq API (llama-3.1-8b-instant) forms natural sentences from detected signs
- **Multi-Language Voice**: gTTS generates speech in 8 languages:
  - English (US, UK, AU accents)
  - Hindi, Spanish, French, German, Kannada
- **Interactive Web UI**: Modern, accessible interface with:
  - Live hand skeleton visualization on canvas overlay
  - Real-time confidence scoring and word chips
  - Sparkline confidence history (24-frame rolling buffer)
  - Gap countdown indicator for sentence formation
  - Voice waveform analyzer with frequency spectrum bars
- **Low-Latency Communication**: Flask-SocketIO with threading for non-blocking frame processing
- **Accessibility-First Design**: Designed for deaf/hard-of-hearing users with clear visual feedback

---

## 🏗️ Architecture

### Backend (Flask-SocketIO)
```
Webcam Input (Browser)
    ↓
Frame Capture (WebRTC/Canvas)
    ↓
Hand Landmark Extraction (MediaPipe)
    ↓
LSTM Prediction (30-frame sliding window, SLIDE_STEP=3)
    ↓
Segmentation State Machine (debounce, word gaps, sentence gaps)
    ↓
Groq LLM Sentence Formation
    ↓
gTTS Audio Generation (background thread)
    ↓
Web Audio API Playback (Browser)
```

### Key Technical Stack
- **Hand Detection**: MediaPipe Hands (21 landmarks × 3 coords = 63 features per frame)
- **Sign Recognition**: LSTM (Keras/TensorFlow)
  - Input: 30-frame sequences (normalized by StandardScaler)
  - Output: Sign probabilities (confidence threshold: 0.60)
- **Sentence Building**: Groq API (temperature=0.2) with ASL-aware prompting
- **Text-to-Speech**: gTTS (Google Translate TTS)
- **Web Server**: Flask + Flask-SocketIO (threading async_mode)
- **Frontend**: Vanilla JavaScript, Web Audio API, Canvas 2D

---

## 📊 Model Details

### Training Data Format
```
real_asl_data/
├── HELLO/
│   ├── HELLO_video_00.npy  (30 frames × 63 features)
│   ├── HELLO_video_01.npy
│   └── ...
├── BOOK/
├── DRINK/
├── CHAIR/
└── ... (more signs)
```

Each `.npy` file contains a sequence of hand landmarks:
- **30 frames** per gesture (capture ~3 seconds of signing at 10fps)
- **21 landmarks** per frame (hand skeleton joints)
- **3D coordinates** (x, y, z) per landmark
- **Total**: 30 × 21 × 3 = 1890 values per sequence

### Model Training
```bash
python training.py  # Train LSTM on custom ASL data
```

Generates:
- `gesture_model.h5` — Trained LSTM model
- `gesture_scaler.pkl` — StandardScaler for feature normalization
- `gesture_labels.pkl` — Sign-to-index mapping

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam (for hand detection)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Dependencies:
- `tensorflow>=2.10.0` — Deep learning framework
- `keras>=2.10.0` — Neural network API
- `opencv-python>=4.6.0` — Computer vision
- `mediapipe>=0.8.9` — Hand landmark detection
- `numpy>=1.21.0` — Numerical computing
- `scikit-learn>=1.0.0` — Data scaling & preprocessing
- `flask>=2.0.0` — Web framework
- `flask-socketio>=5.0.0` — WebSocket communication
- `python-socketio>=5.0.0` — SocketIO server
- `gtts>=2.2.4` — Google Text-to-Speech
- `groq>=0.4.0` — Groq LLM API

### Step 2: Prepare Model Files
Place the trained model files in the project root:
```
gesture_model.h5       — Trained LSTM model
gesture_scaler.pkl     — Feature scaler
gesture_labels.pkl     — Sign labels
```

### Step 3: Set Groq API Key
Export your Groq API key:
```bash
# Linux/Mac
export GROQ_API_KEY="your-api-key-here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your-api-key-here"

# Windows (CMD)
set GROQ_API_KEY=your-api-key-here
```

Get a free API key at https://groq.com

### Step 4: Run the Application
```bash
python app.py
```

The server starts at `http://localhost:5000`

Open your browser and grant **camera access** when prompted.

---

## 🎮 How to Use

### Real-Time Detection
1. **Start the app** — `python app.py`
2. **Open browser** — Navigate to `http://localhost:5000`
3. **Allow camera access** — Click "Allow" on the permission prompt
4. **Sign in front of the webcam** — The system detects your hands in real-time
5. **Watch the UI** — See confidence scores, word chips, and sparkline history
6. **Automatic speech** — When a sentence is complete (4.5-second silence), the system speaks it aloud

### Manual Controls
- **Speak Now** button (or SPACE key) — Manually trigger sentence speech
- **Language dropdown** — Change TTS language (and all future output)
- **Clear button** — Reset the word buffer and start fresh

---

## ⚙️ Configuration

All settings are in `app.py`:

```python
WINDOW_SIZE         = 30      # Frames per sign sequence
SLIDE_STEP          = 3       # Frame advance per prediction (overlap = 27 frames)
CONFIDENCE_THRESH   = 0.60    # Minimum sign confidence to commit
DEBOUNCE_FRAMES     = 4       # Frames to confirm a sign (debounce noise)
WORD_GAP_FRAMES     = 10      # Frames (1 sec) before word boundary
SENTENCE_GAP_FRAMES = 45      # Frames (4.5 sec) before auto-flush
MAX_SENTENCE_WORDS  = 12      # Max words before forced flush
MIN_FLUSH_WORDS     = 1       # Min words to trigger flush
```

### Key Settings Explained

| Setting | Default | Purpose |
|---------|---------|---------|
| `WINDOW_SIZE` | 30 | Input frames for one LSTM prediction |
| `SLIDE_STEP` | 3 | Frame skip between predictions (smaller = more predictions) |
| `CONFIDENCE_THRESH` | 0.60 | Ignore signs below this probability |
| `DEBOUNCE_FRAMES` | 4 | Wait 4 frames before committing a sign (removes flicker) |
| `WORD_GAP_FRAMES` | 10 | 1-second silence before starting a new word |
| `SENTENCE_GAP_FRAMES` | 45 | 4.5-second silence before auto-speaking sentence |
| `MAX_SENTENCE_WORDS` | 12 | Safety limit to prevent runaway sentences |

---

## 📱 Frontend Features

### Live Visualization
- **Hand Skeleton Canvas** — Real-time hand landmark drawing with joint connections
- **Confidence Badge** — Shows sign confidence % on detected words
- **Word Chips** — Badges appear as words are detected, color-coded by confidence
- **Sparkline Chart** — 24-frame rolling window of confidence history
- **Gap Countdown** — Text indicator when system is about to speak ("Speaking soon…")
- **Voice Waveform** — Frequency spectrum analyzer bars sync with audio output

### Responsive Design
- Mobile-friendly layout
- Touch-friendly buttons
- Dark mode with high contrast for accessibility
- Animations on detection and speech events

### Error Handling
- Detects camera permission errors (NotAllowedError, NotFoundError, NotReadableError)
- Specific messaging for IP vs. localhost issues
- Socket.IO connection status indicator ("Live" badge)
- Graceful fallbacks for browser incompatibilities

---

## 🔄 Segmentation Logic

The system uses a **three-state machine** to decide when to form and speak sentences:

```
State          Trigger              Action
─────────────────────────────────────────────────────
Signing        Hand detected +      Add word to buffer,
               confidence > 0.60    reset word gap counter

Word Gap       Hand present but     Increment counter,
               no high-confidence   wait for new sign or
               prediction           timeout

Silence        No hand detected     Count frames without
               for 4.5 seconds      hand; when >= 45 frames,
                                   form sentence & speak
```

**Debounce Mechanism**: Each sign is held for 4 frames before committing to prevent noise from producing spurious words.

**Gap Detection**: The system distinguishes between:
- Short pauses (user thinking) — continue accumulating signs
- Long pauses (end of thought) — form and speak the sentence

---

## 🌍 Language Support

Switch languages in the UI dropdown. Each language generates native-accented speech:

| Code | Language | Voice |
|------|----------|-------|
| `en` | English (US) | Google US English |
| `en-gb` | English (UK) | Google UK English |
| `en-au` | English (AU) | Google Australian English |
| `hi` | Hindi | Google Hindi |
| `es` | Spanish | Google Spanish |
| `fr` | French | Google French |
| `de` | German | Google German |
| `kn` | Kannada | Google Kannada |

---

## 🧠 Training Custom Signs

To add new signs to the model:

### 1. Record Gestures
```bash
python training.py
```
- Press `q` to start recording
- Hold the sign steady for ~3 seconds (30 frames)
- Press `q` again to stop
- Repeat 20+ times per sign for good accuracy

### 2. Retrain the Model
The training script automatically retrains on all samples in `real_asl_data/`.

### 3. Check Performance
```bash
python testing.py
```

---

## 🐛 Troubleshooting

### Camera Access Denied
**Problem**: "Camera access denied" error on page load
**Solution**:
- Check browser permissions (Settings → Privacy → Camera)
- Allow localhost:5000 access
- For IP-based access, use HTTPS or check CORS settings

### No Sign Detection
**Problem**: Signs aren't being recognized
**Possible causes**:
- Model not trained on your signs yet
- Hand position differs from training data (lighting, angle, distance)
- Confidence threshold too high (lower `CONFIDENCE_THRESH`)

**Solution**:
- Train more samples of each sign
- Record training data under similar lighting/angle as testing
- Adjust `CONFIDENCE_THRESH` in `app.py`

### Multiple Flushes Happening
**Problem**: Sentence speaks multiple times in a row
**Solution**: This was a threading race condition. Ensure `frame_lock` is enabled in the latest version.

### Language Not Changing
**Problem**: Changing language in dropdown doesn't affect speech
**Solution**: Ensure socket is connected (check "Live" badge). Refresh page if needed.

### Audio Echo or Distortion
**Problem**: Hearing duplicate or overlapping audio
**Solution**: Audio is cleaned up in the latest version. Refresh browser cache.

---

## 📂 Project Structure

```
SignLanguageDetectionUsingCV/
│
├── app.py                      # Flask-SocketIO backend (main server)
├── sentence_builder.py         # Groq LLM integration for sentence formation
├── training.py                 # Train LSTM model on new signs
├── testing.py                  # Test model accuracy
├── index.html                  # Web UI (frontend)
│
├── gesture_model.h5            # Trained LSTM model
├── gesture_scaler.pkl          # Feature normalization (StandardScaler)
├── gesture_labels.pkl          # Sign-to-index mapping
│
├── real_asl_data/              # Training dataset
│   ├── HELLO/
│   ├── BOOK/
│   ├── DRINK/
│   ├── CHAIR/
│   └── ... (other signs)
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Performance & Optimization

### Inference Speed
- **Frame processing**: ~50-100ms per frame (depends on hand complexity)
- **SocketIO latency**: <10ms (local network)
- **LLM call**: 1-3 seconds (Groq API)
- **TTS generation**: 1-2 seconds (gTTS)

### Memory Usage
- **Model**: ~5MB (LSTM)
- **Scaler**: ~1KB
- **Per-frame buffer**: ~30KB (30 frames × 63 features)

### Optimization Tips
1. **Reduce WINDOW_SIZE** (e.g., 20 instead of 30) for faster inference
2. **Increase SLIDE_STEP** (e.g., 5 instead of 3) to skip more frames
3. **Raise CONFIDENCE_THRESH** (e.g., 0.70) to reduce false positives
4. **Lower MAX_SENTENCE_WORDS** to force early flushing

---

## 🤝 Contributing

To improve the system:

1. **Train more data** — Larger, more diverse datasets improve accuracy
2. **Tune parameters** — Adjust thresholds in `app.py` for your use case
3. **Add new signs** — Run `training.py` to record and integrate new gestures
4. **Improve UI** — Enhance `index.html` for better accessibility

---

## 📋 Roadmap

Future enhancements:
- [ ] Multi-hand detection (fingerspelling support)
- [ ] Offline mode (local LLM instead of Groq API)
- [ ] Desktop/mobile app packaging
- [ ] Sign language corpus expansion
- [ ] Real-time model retraining without server restart
- [ ] User-specific gesture profiles
- [ ] Gesture video recording/playback for verification
- [ ] Integration with accessibility tools (screen readers, etc.)

---

## ⚠️ Limitations

- **Single-hand focus** — Currently optimized for one-handed signs
- **Speed-dependent** — Very fast or very slow signing may not be recognized
- **Lighting-sensitive** — Works best in well-lit environments
- **Training-dependent** — Model accuracy depends on dataset quality and diversity
- **Language coverage** — Limited to 8 languages (extendable via gTTS)

---

## 📄 License

This project is provided as-is for educational and accessibility purposes.

---

## 📞 Support

For issues or questions:
- Check the **Troubleshooting** section above
- Review `app.py` comments for configuration details
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Test with `python testing.py` to verify model

---

## 🙏 Acknowledgments

Built with:
- **MediaPipe** — Hand landmark detection
- **TensorFlow/Keras** — Neural network framework
- **Groq** — Fast LLM API
- **gTTS** — Accessible text-to-speech
- **Flask-SocketIO** — Real-time web communication

Designed for accessibility and real-time performance.

---

**HANDSPEAK** — Making sign language audible, one gesture at a time. 🤲
