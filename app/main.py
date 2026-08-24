from fastapi import FastAPI, UploadFile, File
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import joblib
import soundfile as sf
import numpy as np
import librosa
import io

app = FastAPI()

print("Loading Whisper...")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

print("Loading intent classifier...")
clf = joblib.load("model/saved/intent_classifier.pkl")
vectorizer = joblib.load("model/saved/vectorizer.pkl")

print("Models loaded. API ready.\n")


def run_pipeline(audio_array, sampling_rate):
    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000

    input_features = processor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

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