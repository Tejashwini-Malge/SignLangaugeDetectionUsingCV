"""
Sign Language Translator UI
Real-time sign detection + translation + TTS
Streamlit web interface
"""

import streamlit as st
import cv2
import numpy as np
from collections import deque
import pickle
import time
from tensorflow.keras.models import load_model
import mediapipe as mp
from sign_translater import SignTranslator
from sign_detector import SignAccumulator, SignToSpeechConverter

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤝 Real-Time Sign Language Translator")
st.markdown("Convert sign language to text and speech")

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    """Load model and scaler"""
    try:
        model = load_model('real_asl_model.h5')
        scaler = pickle.load(open('real_asl_scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_resource
def initialize_mediapipe():
    """Initialize MediaPipe"""
    
    mp_hands = mp.solutions.hands
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # FIXED
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    return hands, mp_hands

@st.cache_resource
def get_translator():
    return SignTranslator()

@st.cache_resource
def get_tts_converter():
    return SignToSpeechConverter(language='en', slow=False)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("⚙️ Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.5,
    max_value=0.99,
    value=0.85,
    step=0.01
)

pause_duration = st.sidebar.slider(
    "Pause Duration (seconds)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1
)

auto_play_tts = st.sidebar.checkbox(
    "Auto-play TTS",
    value=True
)

show_landmarks = st.sidebar.checkbox(
    "Show Hand Landmarks",
    value=True
)

# ==========================================
# LOAD TRANSLATOR
# ==========================================

translator = get_translator()

st.sidebar.markdown("---")
st.sidebar.markdown("### Supported Signs")

try:
    signs_list = translator.get_vocabulary()
    st.sidebar.write(", ".join(signs_list))
except:
    st.sidebar.write("Vocabulary not available")

# ==========================================
# MAIN LAYOUT
# ==========================================

col1, col2 = st.columns([2, 1])

# ==========================================
# LEFT PANEL
# ==========================================

with col1:
    
    st.subheader("📹 Live Detection")
    
    model, scaler = load_models()
    hands, mp_hands = initialize_mediapipe()
    
    if model is None or scaler is None:
        st.stop()
    
    # Session state
    if 'accumulator' not in st.session_state:
        st.session_state.accumulator = SignAccumulator(
            confidence_threshold=confidence_threshold,
            pause_duration=pause_duration
        )
    
    if 'translations_history' not in st.session_state:
        st.session_state.translations_history = []
    
    # Webcam placeholders
    stframe = st.empty()
    status_placeholder = st.empty()
    
    # Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot access webcam")
    
    else:
        
        col_start, col_clear = st.columns(2)
        
        with col_start:
            run_detection = st.button(
                "🎥 Start Detection",
                use_container_width=True
            )
        
        with col_clear:
            clear_history = st.button(
                "🗑️ Clear History",
                use_container_width=True
            )
        
        # Clear history
        if clear_history:
            st.session_state.translations_history = []
            st.session_state.accumulator.clear_buffer()
        
        # ==========================================
        # START DETECTION
        # ==========================================
        
        if run_detection:
            
            st.info("Detection started...")
            
            accumulator = st.session_state.accumulator
            translator = get_translator()
            tts = get_tts_converter()
            
            # FIXED BUFFER
            frame_buffer = deque(maxlen=30)
            
            while True:
                
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("Failed to read webcam frame")
                    break
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hands
                results = hands.process(rgb_frame)
                
                # ==========================================
                # HAND DETECTED
                # ==========================================
                
                if results.multi_hand_landmarks:
                    
                    # ONLY USE FIRST HAND
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    landmarks = []
                    
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # ENSURE EXACTLY 63 VALUES
                    if len(landmarks) == 63:
                        frame_buffer.append(np.array(landmarks))
                    else:
                        frame_buffer.append(np.zeros(63))
                    
                    # Draw landmarks
                    if show_landmarks:
                        
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                    
                    # ==========================================
                    # PREDICTION
                    # ==========================================
                    
                    if len(frame_buffer) == 30:
                        
                        try:
                            
                            # Convert buffer to array
                            X = np.array(frame_buffer)
                            
                            # Shape check
                            X = X.reshape(1, 30, 63)
                            
                            # Scale data
                            X_flat = X.reshape(-1, 63)
                            X_flat = scaler.transform(X_flat)
                            X = X_flat.reshape(1, 30, 63)
                            
                            # Predict
                            pred = model.predict(X, verbose=0)
                            
                            confidence = float(np.max(pred))
                            pred_idx = int(np.argmax(pred))
                            
                            # Get sign name
                            sign_names = list(translator.sign_to_word.keys())
                            
                            if pred_idx < len(sign_names):
                                sign_name = sign_names[pred_idx]
                            else:
                                sign_name = "Unknown"
                            
                            # Add prediction
                            should_translate, buffer = accumulator.add_prediction(
                                sign_name,
                                confidence
                            )
                            
                            # Show prediction
                            status_text = f"{sign_name} ({confidence:.2f})"
                            
                            cv2.putText(
                                frame,
                                status_text,
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )
                            
                            # ==========================================
                            # TRANSLATE
                            # ==========================================
                            
                            if should_translate:
                                
                                signs_sequence = accumulator.get_buffer()
                                
                                if signs_sequence:
                                    
                                    # Translate
                                    text = translator.translate(signs_sequence)
                                    
                                    # Save history
                                    st.session_state.translations_history.append({
                                        "signs": " → ".join(signs_sequence),
                                        "text": text,
                                        "timestamp": time.time()
                                    })
                                    
                                    # Speak
                                    if auto_play_tts:
                                        tts.speak(text, auto_play=True)
                                    
                                    # Clear buffer
                                    accumulator.clear_buffer()
                        
                        except Exception as e:
                            st.error(f"Prediction Error: {str(e)}")
                
                # ==========================================
                # NO HAND DETECTED
                # ==========================================
                
                else:
                    
                    cv2.putText(
                        frame,
                        "No Hands Detected",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                
                # ==========================================
                # DISPLAY FRAME
                # ==========================================
                
                stframe.image(
                    frame,
                    channels="BGR",
                    width="stretch"
                )
                
                # Status
                status_placeholder.info(
                    f"Detected Signs: {accumulator.detected_signs_count} | "
                    f"Translations: {accumulator.translations_count}"
                )

# ==========================================
# RIGHT PANEL
# ==========================================

with col2:
    
    st.subheader("📝 Translations")
    
    if st.session_state.translations_history:
        
        for i, item in enumerate(
            reversed(st.session_state.translations_history[-10:])
        ):
            
            with st.container(border=True):
                
                st.write(f"**Signs:** {item['signs']}")
                st.write(f"**Text:** {item['text']}")
                
                st.caption(
                    f"Time: "
                    f"{time.strftime('%H:%M:%S', time.localtime(item['timestamp']))}"
                )
                
                if st.button(f"🔊 Repeat", key=f"speak_{i}"):
                    
                    tts = get_tts_converter()
                    tts.speak(item['text'], auto_play=True)
    
    else:
        st.info("No translations yet")

# ==========================================
# CLEANUP
# ==========================================

if 'cap' in locals():
    cap.release()

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
### How to Use

1. Allow webcam access
2. Click **Start Detection**
3. Show trained signs clearly
4. Wait briefly between signs
5. Translation and speech will appear automatically

### Tips
- Use good lighting
- Keep hand visible
- Avoid fast movement
- Use one hand only

### Supported Signs
HELLO, LOVE, HELP, BOOK, CHAIR, DRINK, BEAUTIFUL
""")