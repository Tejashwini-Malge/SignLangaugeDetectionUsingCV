"""
REAL-TIME ASL SIGN DETECTION — SENTENCE MODE
- Sliding window inference (continuous, no manual trigger)
- Segmentation via hand-gap + confidence drop
- Sentence buffer accumulation
- TTS voice output via pyttsx3

Install: pip install pyttsx3
"""

import numpy as np
import cv2
import mediapipe as mp
import pickle
import threading
import time
from collections import deque
from tensorflow.keras.models import load_model

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("[WARNING] pyttsx3 not installed. Run: pip install pyttsx3")
    TTS_AVAILABLE = False


# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_PATH   = 'real_asl_model.h5'
SCALER_PATH  = 'real_asl_scaler.pkl'
SIGNS_PATH   = 'real_asl_signs.pkl'

WINDOW_SIZE        = 30     # frames fed to LSTM
SLIDE_STEP         = 5      # run inference every N frames
CONFIDENCE_THRESH  = 0.80   # minimum confidence to accept a prediction
WORD_GAP_FRAMES    = 15     # no-hand frames = word boundary
SENTENCE_GAP_FRAMES= 45     # no-hand frames = end of sentence (~1.5 sec at 30fps)
MAX_SENTENCE_WORDS = 12     # auto-flush sentence after this many words
DEBOUNCE_FRAMES    = 10     # same word must appear this many frames before accepted


# ─── TTS THREAD ────────────────────────────────────────────────────────────────

class TTSSpeaker:
    """Runs TTS in a background thread so it never blocks the camera loop."""

    def __init__(self):
        self._queue = []
        self._lock  = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def speak(self, text):
        with self._lock:
            self._queue.append(text)

    def _run(self):
        if not TTS_AVAILABLE:
            return
        engine = pyttsx3.init()
        engine.setProperty('rate', 155)   # words per minute
        engine.setProperty('volume', 1.0)
        while True:
            with self._lock:
                if self._queue:
                    text = self._queue.pop(0)
                else:
                    text = None
            if text:
                print(f"\n🔊  Speaking: \"{text}\"")
                engine.say(text)
                engine.runAndWait()
            else:
                time.sleep(0.05)


# ─── SEGMENTATION STATE MACHINE ────────────────────────────────────────────────

class SegmentationState:
    """
    Tracks word boundaries and sentence boundaries.

    Word boundary  → hand absent for WORD_GAP_FRAMES
    Sentence end   → hand absent for SENTENCE_GAP_FRAMES
    """

    def __init__(self, speaker: TTSSpeaker):
        self.speaker          = speaker
        self.sentence_buffer  = []          # list of accepted words
        self.no_hand_frames   = 0
        self.last_committed   = None        # last word added to buffer
        self.debounce_counter = 0           # consecutive frames with same prediction
        self.debounce_word    = None
        self.word_boundary_hit= False       # True after a gap, reset on next detection

    # called every frame --------------------------------------------------------

    def update(self, hand_detected: bool, predicted_word: str | None, confidence: float):
        """
        hand_detected : bool
        predicted_word: str or None (None if confidence below threshold)
        confidence    : float 0-1
        """

        if hand_detected and predicted_word is not None:
            self.no_hand_frames    = 0
            self.word_boundary_hit = False
            self._try_commit(predicted_word, confidence)

        else:
            self.no_hand_frames   += 1
            self.debounce_counter  = 0
            self.debounce_word     = None

            if self.no_hand_frames == WORD_GAP_FRAMES:
                # Word boundary — allow the next different word to be committed
                self.word_boundary_hit = True
                self.last_committed    = None   # reset so same sign can repeat after gap

            if self.no_hand_frames == SENTENCE_GAP_FRAMES:
                self._flush_sentence()

        # Safety flush on max length
        if len(self.sentence_buffer) >= MAX_SENTENCE_WORDS:
            self._flush_sentence()

    def _try_commit(self, word: str, confidence: float):
        """Debounce: require DEBOUNCE_FRAMES consecutive same prediction."""
        if word == self.debounce_word:
            self.debounce_counter += 1
        else:
            self.debounce_word    = word
            self.debounce_counter = 1

        if self.debounce_counter >= DEBOUNCE_FRAMES:
            if word != self.last_committed:
                self.sentence_buffer.append(word)
                self.last_committed    = word
                self.debounce_counter  = 0
                print(f"  ✚ Word committed: {word}  →  buffer: {self.sentence_buffer}")

    def _flush_sentence(self):
        if not self.sentence_buffer:
            return
        sentence = " ".join(self.sentence_buffer)
        print(f"\n{'─'*60}")
        print(f"  📝  Sentence: {sentence}")
        print(f"{'─'*60}\n")
        self.speaker.speak(sentence)
        self.sentence_buffer  = []
        self.last_committed   = None
        self.no_hand_frames   = 0

    def force_flush(self):
        """Called on SPACE key press."""
        self._flush_sentence()

    @property
    def current_sentence(self):
        return " ".join(self.sentence_buffer)


# ─── DRAW HUD ──────────────────────────────────────────────────────────────────

def draw_hud(frame, predicted_word, confidence, state: SegmentationState, top3):
    h, w = frame.shape[:2]

    # ── top bar ──
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    if predicted_word:
        color = (0, 220, 100) if confidence >= CONFIDENCE_THRESH else (0, 140, 220)
        cv2.putText(frame, f"{predicted_word}  {confidence:.0%}",
                    (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)
    else:
        cv2.putText(frame, "— waiting —",
                    (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 1)

    # ── sentence strip ──
    sentence_text = state.current_sentence if state.current_sentence else "..."
    cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, sentence_text,
                (16, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1)
    cv2.putText(frame, "SENTENCE",
                (16, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    # ── top-3 sidebar ──
    cv2.rectangle(frame, (w - 200, 0), (w, 110), (30, 30, 30), -1)
    cv2.putText(frame, "Top predictions",
                (w - 195, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    for i, (sign, conf) in enumerate(top3):
        y = 42 + i * 24
        bar_w = int(160 * conf)
        cv2.rectangle(frame, (w - 195, y - 14), (w - 195 + bar_w, y + 2), (50, 100, 60), -1)
        cv2.putText(frame, f"{sign}  {conf:.0%}",
                    (w - 193, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 200), 1)

    # ── gap indicator ──
    gap = state.no_hand_frames
    if gap > 0:
        ratio   = min(gap / SENTENCE_GAP_FRAMES, 1.0)
        bar_max = w - 40
        bar_w   = int(bar_max * ratio)
        color   = (0, int(200 * (1 - ratio)), int(200 * ratio))
        cv2.rectangle(frame, (20, 75), (20 + bar_w, 84), color, -1)
        cv2.putText(frame, "gap", (20, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    # ── controls ──
    cv2.putText(frame, "SPACE: speak now   Q: quit",
                (16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1)

    return frame


# ─── MAIN DETECTION LOOP ───────────────────────────────────────────────────────

def run_detection():

    # Load model + assets
    print("\nLoading model...")
    try:
        model    = load_model(MODEL_PATH)
        scaler   = pickle.load(open(SCALER_PATH, 'rb'))
        sign_map = pickle.load(open(SIGNS_PATH,  'rb'))
    except FileNotFoundError as e:
        print(f"❌  Missing file: {e}")
        print("    Train the model first: python realdatatraining.py")
        return

    num_classes = len(sign_map)
    print(f"✓  Model loaded — {num_classes} signs: {list(sign_map.values())}")

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    # State
    speaker   = TTSSpeaker()
    seg_state = SegmentationState(speaker)
    frame_buf = deque(maxlen=WINDOW_SIZE)   # rolling landmark buffer
    frame_idx = 0                           # total frames processed
    predicted_word = None
    confidence     = 0.0
    top3           = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam.")
        return

    print("\n" + "="*60)
    print("  REAL-TIME SENTENCE DETECTION STARTED")
    print("  Sign slowly and pause between words.")
    print("  SPACE = speak sentence now   Q = quit")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_frame)
        frame_idx += 1

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            lm = results.multi_hand_landmarks[0]

            # Draw skeleton
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Collect 63-d landmark vector
            hand_vec = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            frame_buf.append(hand_vec)

            # Run inference every SLIDE_STEP frames once buffer is full
            if len(frame_buf) == WINDOW_SIZE and frame_idx % SLIDE_STEP == 0:
                seq = np.array(frame_buf)           # (30, 63)
                seq = scaler.transform(seq)
                seq = seq.reshape(1, WINDOW_SIZE, 63)

                probs     = model.predict(seq, verbose=0)[0]
                top_idx   = np.argsort(probs)[::-1]
                top3      = [(sign_map[i], probs[i]) for i in top_idx[:3]]
                sign_id   = top_idx[0]
                confidence= probs[sign_id]

                if confidence >= CONFIDENCE_THRESH:
                    predicted_word = sign_map[sign_id]
                else:
                    predicted_word = None

        else:
            predicted_word = None
            confidence     = 0.0

        # Update segmentation state machine
        seg_state.update(hand_detected, predicted_word, confidence)

        # Draw HUD
        frame = draw_hud(frame, predicted_word, confidence, seg_state, top3)
        cv2.imshow("ASL Sentence Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            seg_state.force_flush()

    # Flush any remaining words on exit
    seg_state.force_flush()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✓  Detection stopped.")


# ─── ENTRY ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_detection()