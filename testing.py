"""
REAL-TIME ASL SIGN DETECTION — SENTENCE MODE + WEBSOCKET SERVER
- Sliding window inference
- Segmentation via hand-gap + confidence
- Sentence buffer + TTS (pyttsx3)
- WebSocket server on ws://localhost:8765 → streams predictions to browser UI

Install: pip install pyttsx3 websockets
Run:     python trainingrealdata.py
Then open index.html in your browser.
"""

import numpy as np
import cv2
import mediapipe as mp
import pickle
import threading
import time
import json
import asyncio
import websockets
from collections import deque
from tensorflow.keras.models import load_model

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("[WARNING] pyttsx3 not installed. Run: pip install pyttsx3")
    TTS_AVAILABLE = False


# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_PATH          = 'real_asl_model.h5'
SCALER_PATH         = 'real_asl_scaler.pkl'
SIGNS_PATH          = 'real_asl_signs.pkl'

WINDOW_SIZE         = 30
SLIDE_STEP          = 5
CONFIDENCE_THRESH   = 0.60
WORD_GAP_FRAMES     = 15
SENTENCE_GAP_FRAMES = 45
MAX_SENTENCE_WORDS  = 12
DEBOUNCE_FRAMES     = 10

WS_HOST             = 'localhost'
WS_PORT             = 8765


# ─── WEBSOCKET BROADCASTER ─────────────────────────────────────────────────────

class WSBroadcaster:
    """
    Runs an asyncio WebSocket server in a background thread.
    The main OpenCV loop calls broadcast() to send data to all connected clients.
    """

    def __init__(self):
        self._clients   = set()
        self._loop      = asyncio.new_event_loop()
        self._lock      = threading.Lock()
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, WS_HOST, WS_PORT):
            print(f"✓  WebSocket server running on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()   # run forever

    async def _handler(self, ws):
        with self._lock:
            self._clients.add(ws)
        print(f"  Browser connected ({len(self._clients)} client(s))")
        try:
            await ws.wait_closed()
        finally:
            with self._lock:
                self._clients.discard(ws)

    def broadcast(self, payload: dict):
        """Called from the main thread — thread-safe send to all clients."""
        msg = json.dumps(payload)
        with self._lock:
            clients = set(self._clients)
        if not clients:
            return
        async def _send_all():
            await asyncio.gather(
                *[c.send(msg) for c in clients],
                return_exceptions=True
            )
        asyncio.run_coroutine_threadsafe(_send_all(), self._loop)


# ─── TTS THREAD ────────────────────────────────────────────────────────────────

class TTSSpeaker:
    def __init__(self):
        self._queue  = []
        self._lock   = threading.Lock()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def speak(self, text):
        with self._lock:
            self._queue.append(text)

    def _run(self):
        if not TTS_AVAILABLE:
            return
        engine = pyttsx3.init()
        engine.setProperty('rate', 155)
        engine.setProperty('volume', 1.0)
        while True:
            with self._lock:
                text = self._queue.pop(0) if self._queue else None
            if text:
                print(f"\n🔊  Speaking: \"{text}\"")
                engine.say(text)
                engine.runAndWait()
            else:
                time.sleep(0.05)


# ─── SEGMENTATION STATE MACHINE ────────────────────────────────────────────────

class SegmentationState:
    def __init__(self, speaker: TTSSpeaker, broadcaster: WSBroadcaster):
        self.speaker          = speaker
        self.broadcaster      = broadcaster
        self.sentence_buffer  = []
        self.no_hand_frames   = 0
        self.last_committed   = None
        self.debounce_counter = 0
        self.debounce_word    = None

    def update(self, hand_detected: bool, predicted_word, confidence: float):
        if hand_detected and predicted_word is not None:
            self.no_hand_frames = 0
            self._try_commit(predicted_word, confidence)
        else:
            self.no_hand_frames  += 1
            self.debounce_counter = 0
            self.debounce_word    = None
            if self.no_hand_frames == WORD_GAP_FRAMES:
                self.last_committed = None
            if self.no_hand_frames == SENTENCE_GAP_FRAMES:
                self._flush_sentence()

        if len(self.sentence_buffer) >= MAX_SENTENCE_WORDS:
            self._flush_sentence()

    def _try_commit(self, word: str, confidence: float):
        if word == self.debounce_word:
            self.debounce_counter += 1
        else:
            self.debounce_word    = word
            self.debounce_counter = 1

        if self.debounce_counter >= DEBOUNCE_FRAMES and word != self.last_committed:
            self.sentence_buffer.append(word)
            self.last_committed   = word
            self.debounce_counter = 0
            print(f"  ✚ Word: {word}  →  {self.sentence_buffer}")

    def _flush_sentence(self):
        if not self.sentence_buffer:
            return
        sentence = " ".join(self.sentence_buffer)
        print(f"\n  📝  Sentence: {sentence}\n")
        self.speaker.speak(sentence)
        # tell browser to flush
        self.broadcaster.broadcast({"event": "flush", "sentence": sentence})
        self.sentence_buffer  = []
        self.last_committed   = None
        self.no_hand_frames   = 0

    def force_flush(self):
        self._flush_sentence()

    @property
    def current_sentence(self):
        return " ".join(self.sentence_buffer)


# ─── MAIN DETECTION LOOP ───────────────────────────────────────────────────────

def run_detection():
    print("\nLoading model...")
    try:
        model    = load_model(MODEL_PATH)
        scaler   = pickle.load(open(SCALER_PATH, 'rb'))
        sign_map = pickle.load(open(SIGNS_PATH,  'rb'))
    except FileNotFoundError as e:
        print(f"❌  Missing file: {e}")
        return

    print(f"✓  Model loaded — {len(sign_map)} signs: {list(sign_map.values())}")

    # Start WebSocket server
    broadcaster = WSBroadcaster()
    time.sleep(0.5)   # let server boot

    # TTS
    speaker   = TTSSpeaker()
    seg_state = SegmentationState(speaker, broadcaster)

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    frame_buf      = deque(maxlen=WINDOW_SIZE)
    frame_idx      = 0
    predicted_word = None
    confidence     = 0.0
    top3           = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open webcam.")
        return

    print("\n" + "="*60)
    print("  DETECTION STARTED")
    print(f"  Open index.html in your browser")
    print(f"  WebSocket: ws://{WS_HOST}:{WS_PORT}")
    print("  SPACE = speak now   Q = quit")
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
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            hand_vec = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            frame_buf.append(hand_vec)

            if len(frame_buf) == WINDOW_SIZE and frame_idx % SLIDE_STEP == 0:
                seq   = np.array(frame_buf)
                seq   = scaler.transform(seq)
                seq   = seq.reshape(1, WINDOW_SIZE, 63)

                probs     = model.predict(seq, verbose=0)[0]
                top_idx   = np.argsort(probs)[::-1]
                top3      = [(sign_map[i], float(probs[i])) for i in top_idx[:3]]
                sign_id   = top_idx[0]
                confidence= float(probs[sign_id])

                predicted_word = sign_map[sign_id] if confidence >= CONFIDENCE_THRESH else None

                # ── broadcast to browser ──
                broadcaster.broadcast({
                    "event":      "prediction",
                    "word":       predicted_word,
                    "confidence": confidence,
                    "top3":       top3,
                    "sentence":   seg_state.current_sentence,
                    "no_hand":    seg_state.no_hand_frames,
                    "hand":       True
                })

        else:
            predicted_word = None
            confidence     = 0.0
            # broadcast no-hand state
            broadcaster.broadcast({
                "event":      "prediction",
                "word":       None,
                "confidence": 0.0,
                "top3":       top3,
                "sentence":   seg_state.current_sentence,
                "no_hand":    seg_state.no_hand_frames,
                "hand":       False
            })

        seg_state.update(hand_detected, predicted_word, confidence)

        # minimal OpenCV window — just the raw camera feed with skeleton
        cv2.imshow("Handspeak — camera (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            seg_state.force_flush()

    seg_state.force_flush()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✓  Detection stopped.")


if __name__ == "__main__":
    run_detection()