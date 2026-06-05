from gtts import gTTS
import io, base64

print("generating...")
tts = gTTS(text="hello this is a test", lang='en', slow=False)
buf = io.BytesIO()
tts.write_to_fp(buf)
buf.seek(0)
audio_b64 = base64.b64encode(buf.read()).decode('utf-8')
print(f"audio length: {len(audio_b64)}")

with open('test_output.mp3', 'wb') as f:
    f.write(base64.b64decode(audio_b64))
print("saved test_output.mp3 — open it and check if it plays")