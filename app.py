"""
HANDSPEAK — Flask-SocketIO Backend
async_mode=threading — works with TensorFlow + OpenCV on Windows, no monkey patching

Install:
    pip install flask flask-socketio mediapipe tensorflow opencv-python-headless gtts numpy groq

Run:
    python app.py

Open:
    http://localhost:5000
"""

# NO eventlet, NO gevent, NO monkey patching
import cv2
import numpy as np
import pickle
import base64
import io
import threading
from collections import deque
from sentence_builder import build_sentence

from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit

import mediapipe as mp
from tensorflow.keras.models import load_model
from gtts import gTTS

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_PATH          = 'real_asl_model.h5'
SCALER_PATH         = 'real_asl_scaler.pkl'
SIGNS_PATH          = 'real_asl_signs.pkl'

WINDOW_SIZE         = 30
SLIDE_STEP          = 3
CONFIDENCE_THRESH   = 0.60
DEBOUNCE_FRAMES     = 4
WORD_GAP_FRAMES     = 10
SENTENCE_GAP_FRAMES = 45   # 4.5 sec at 10fps — gives time to think between signs
MAX_SENTENCE_WORDS  = 12
MIN_FLUSH_WORDS     = 1    # raise to 2 to prevent single-word accidental flushes

# MediaPipe hand landmark connections for skeleton drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm
]

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────

print("\nLoading model...")
model    = load_model(MODEL_PATH)
scaler   = pickle.load(open(SCALER_PATH, 'rb'))
sign_map = pickle.load(open(SIGNS_PATH,  'rb'))
print(f"✓ Model loaded — {len(sign_map)} signs: {list(sign_map.values())}")

mp_hands_mod = mp.solutions.hands

# ─── TTS ───────────────────────────────────────────────────────────────────────

# gTTS language map: key = dropdown value → (lang_code, tld or None)
# tld only applies to English — other languages use plain lang code
TTS_LANG_MAP = {
    'en':    ('en', 'com'),    # English US
    'en-gb': ('en', 'co.uk'), # English UK  (accent via tld)
    'en-au': ('en', 'com.au'),# English AU  (accent via tld)
    'hi':    ('hi', None),    # Hindi
    'es':    ('es', None),    # Spanish
    'fr':    ('fr', None),    # French
    'de':    ('de', None),    # German
    'kn':    ('kn', None),    # Kannada
}

def tts_to_base64(text, lang="en"):
    lang_code, tld = TTS_LANG_MAP.get(lang, ('en', 'com'))
    try:
        print(f"Generating TTS [{lang_code}]: {text}")
        buf = io.BytesIO()
        if tld:
            tts = gTTS(text=text, lang=lang_code, tld=tld, slow=False)
        else:
            tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
        print("TTS ready")
        return audio_b64
    except Exception as e:
        print(f"TTS ERROR: {repr(e)}")
        return None

def emit_flush_async(sid, sentence, lang="en"):
    """Run TTS in a background thread so it doesn't block frame processing."""
    def _worker():
        audio = tts_to_base64(sentence, lang=lang)
        socketio.emit('flush', {'sentence': sentence, 'audio': audio}, to=sid)
    threading.Thread(target=_worker, daemon=True).start()

# ─── SEGMENTATION ──────────────────────────────────────────────────────────────

class SegmentationState:
    def __init__(self):
        self.buffer    = []
        self.no_hand   = 0
        self.last_word = None
        self.deb_count = 0
        self.deb_word  = None

    def update(self, hand_detected, word, confidence):
        if not hand_detected:
            # ── True gap: hand is completely absent ──────────────────────────
            self.no_hand  += 1
            self.deb_count = 0      # reset debounce only on true absence
            self.deb_word  = None
            if self.no_hand == WORD_GAP_FRAMES:
                self.last_word = None
            if self.no_hand >= SENTENCE_GAP_FRAMES:
                return self._flush()
        elif word:
            # ── Hand present AND confident prediction ────────────────────────
            self.no_hand = 0
            self._try_commit(word)
        else:
            # ── Hand present but model uncertain (slide-step or low conf) ────
            # Don't touch debounce or no_hand — just wait for next good frame
            self.no_hand = 0

        if len(self.buffer) >= MAX_SENTENCE_WORDS:
            return self._flush()
        return None

    def _try_commit(self, word):
        if word == self.deb_word:
            self.deb_count += 1
        else:
            self.deb_word  = word
            self.deb_count = 1
        if self.deb_count >= DEBOUNCE_FRAMES and word != self.last_word:
            self.buffer.append(word)
            self.last_word = word
            self.deb_count = 0
            print(f"  ✚ Word: '{word}'  →  {self.buffer}")

    def _flush(self, force=False):
        if not self.buffer:
            return None
        # Don't auto-flush a single word — wait for more signs or manual trigger
        if not force and len(self.buffer) < MIN_FLUSH_WORDS:
            return None
        sentence = build_sentence(self.buffer)
        print(f"  📝 FLUSH: '{sentence}'")
        self.buffer    = []
        self.last_word = None
        self.no_hand   = 0
        self.deb_count = 0
        self.deb_word  = None
        return sentence

    def force_flush(self):
        return self._flush(force=True)

    @property
    def sentence(self):
        return " ".join(self.buffer)

# ─── PER-CLIENT STATE ──────────────────────────────────────────────────────────

clients = {}
clients_lock = threading.Lock()

def get_client(sid):
    with clients_lock:
        if sid not in clients:
            clients[sid] = {
                'seg':        SegmentationState(),
                'frame_buf':  deque(maxlen=WINDOW_SIZE),
                'frame_idx':  0,
                'top3':       [],
                'last_pred':  None,
                'last_conf':  0.0,
                'last_lm':    None,   # last hand landmarks for skeleton
                'tts_lang':   'en',   # TTS language code
                'hands_lock': threading.Lock(),
                'frame_lock': threading.Lock(),  # prevents concurrent frame processing
                'hands':      mp_hands_mod.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.50,
                    min_tracking_confidence=0.50,
                )
            }
        return clients[sid]

# ─── FLASK + SOCKETIO ──────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'handspeak-secret'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=5 * 1024 * 1024
)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

# ─── SOCKET EVENTS ─────────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    sid = request.sid
    get_client(sid)
    print(f"\n── Client connected: {sid[:8]} ──")

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    with clients_lock:
        if sid in clients:
            clients[sid]['hands'].close()
            del clients[sid]
    print(f"── Client disconnected: {sid[:8]} ──\n")

@socketio.on('frame')
def on_frame(data):
    sid    = request.sid
    client = get_client(sid)

    # One frame at a time per client — prevents concurrent flush race conditions
    if not client['frame_lock'].acquire(blocking=False):
        return
    try:
        _process_frame(client, sid, data)
    finally:
        client['frame_lock'].release()

def _process_frame(client, sid, data):
    seg         = client['seg']
    frame_buf   = client['frame_buf']
    hands       = client['hands']
    hands_lock  = client['hands_lock']
    client['frame_idx'] += 1
    frame_idx   = client['frame_idx']

    # ── Decode base64 JPEG ──
    try:
        padding = 4 - len(data) % 4
        if padding != 4:
            data = data + ('=' * padding)
        img_bytes = base64.b64decode(data)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            frame = cv2.imdecode(np.asarray(bytearray(img_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception as e:
        print(f"  [ERROR] frame decode: {e}")
        return

    # ── MediaPipe ──
    rgb           = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_detected = False
    word          = None
    confidence    = 0.0

    try:
        with hands_lock:
            results = hands.process(rgb)
    except ValueError as e:
        print(f"  [MediaPipe] skipped frame: {e}")
        return

    if results.multi_hand_landmarks:
        hand_detected = True
        lm       = results.multi_hand_landmarks[0]
        hand_vec = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
        frame_buf.append(hand_vec)
        client['last_lm'] = [{'x': p.x, 'y': p.y} for p in lm.landmark]

        if len(frame_buf) == WINDOW_SIZE and frame_idx % SLIDE_STEP == 0:
            seq       = np.array(frame_buf)
            seq       = scaler.transform(seq)
            seq       = seq.reshape(1, WINDOW_SIZE, 63)
            probs     = model.predict(seq, verbose=0)[0]
            top_idx   = np.argsort(probs)[::-1]

            client['top3']      = [[sign_map[i], float(probs[i])] for i in top_idx[:3]]
            sign_id             = top_idx[0]
            confidence          = float(probs[sign_id])
            client['last_conf'] = confidence
            if confidence >= CONFIDENCE_THRESH:
                word = sign_map[sign_id]
            client['last_pred'] = word
    else:
        client['last_lm'] = None

    # ── Segmentation ──
    flushed = seg.update(hand_detected, word, confidence)

    if flushed:
        emit_flush_async(sid, flushed, client['tts_lang'])

    emit('prediction', {
        'word':       client['last_pred'],
        'confidence': client['last_conf'],
        'top3':       client['top3'],
        'sentence':   seg.sentence,
        'no_hand':    seg.no_hand,
        'hand':       hand_detected,
        'landmarks':  client['last_lm'],   # hand skeleton points
    })

@socketio.on('flush')
def on_flush(data=None):
    sid    = request.sid
    client = get_client(sid)
    # Accept lang override sent directly with the flush event
    if data and isinstance(data, dict) and 'lang' in data:
        client['tts_lang'] = data['lang']
    s = client['seg'].force_flush()
    if s:
        emit_flush_async(sid, s, client['tts_lang'])
    else:
        print("  Force flush — buffer empty")

@socketio.on('set_lang')
def on_set_lang(data):
    sid    = request.sid
    client = get_client(sid)
    lang   = data.get('lang', 'en')
    client['tts_lang'] = lang
    print(f"  🌐 TTS language set to: {lang}")

# ─── ENTRY ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  HANDSPEAK  —  http://localhost:5000")
    print("="*50 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
