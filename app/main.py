from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import joblib
import soundfile as sf
import numpy as np
import librosa
import io

app = FastAPI()

print("Loading Whisper (faster-whisper, tiny, int8 for low memory)...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("Loading intent classifier...")
clf = joblib.load("model/saved/intent_classifier.pkl")
vectorizer = joblib.load("model/saved/vectorizer.pkl")

print("Models loaded. API ready.\n")


def run_pipeline(audio_array, sampling_rate):
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)

    segments, _ = whisper_model.transcribe(audio_array, language="en")
    transcription = " ".join([segment.text for segment in segments]).strip()

    text_vector = vectorizer.transform([transcription])
    intent = clf.predict(text_vector)[0]

    return {"transcription": transcription, "intent": intent}


@app.get("/")
def health_check():
    return {"status": "AI Voice Receptionist API is running"}


@app.post("/process-call")
async def process_call(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))

    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    audio_array = audio_array.astype(np.float32)

    result = run_pipeline(audio_array, sampling_rate)
    return result