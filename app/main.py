from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import joblib
import soundfile as sf
import numpy as np
import librosa
import io
import requests

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


@app.post("/incoming-call")
async def incoming_call():
    swml_response = {
        "version": "1.0.0",
        "sections": {
            "main": [
                {
                    "play": {
                        "url": "say:Hello, thanks for calling. Please tell me how I can help after the beep."
                    }
                },
                {
                    "record": {
                        "audio": True,
                        "format": "wav",
                        "max_length": 10,
                        "beep": True
                    }
                },
                {
                    "request": {
                        "url": "https://ai-voice-receptionist-xgw2.onrender.com/process-recording",
                        "method": "POST"
                    }
                }
            ]
        }
    }
    return JSONResponse(content=swml_response)


@app.post("/process-recording")
async def process_recording(RecordingUrl: str = None):
    audio_response = requests.get(RecordingUrl + ".wav")
    audio_array, sampling_rate = sf.read(io.BytesIO(audio_response.content))

    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    audio_array = audio_array.astype(np.float32)

    result = run_pipeline(audio_array, sampling_rate)

    swml_response = {
        "version": "1.0.0",
        "sections": {
            "main": [
                {
                    "play": {
                        "url": f"say:I heard you say: {result['transcription']}. I think you want to {result['intent']}. Goodbye."
                    }
                }
            ]
        }
    }
    return JSONResponse(content=swml_response)