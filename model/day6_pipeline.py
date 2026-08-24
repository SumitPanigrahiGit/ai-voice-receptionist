"""
Day 6 — Combined Pipeline (Whisper + Intent Classifier)
Generates its own test audio using offline text-to-speech, resamples it 
to match Whisper's required 16kHz, and runs the full pipeline.
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import joblib
import soundfile as sf
import numpy as np
import librosa
import pyttsx3
import os

# ---- Step 1: Load models ----
print("Loading Whisper...")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

print("Loading intent classifier...")
clf = joblib.load("model/saved/intent_classifier.pkl")
vectorizer = joblib.load("model/saved/vectorizer.pkl")

print("All models loaded.\n")


# ---- Step 2: The pipeline function ----
def process_call(audio_array, sampling_rate):
    # Whisper requires 16kHz audio — resample if needed
    if sampling_rate != 16000:
        audio_array = librosa.resample(
            audio_array, orig_sr=sampling_rate, target_sr=16000
        )
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


# ---- Step 3: Generate test audio using offline text-to-speech ----
os.makedirs("model/test_audio", exist_ok=True)

test_sentences = [
    "I want to book an appointment for next Tuesday",
    "Can I reschedule my appointment to Friday",
    "I need to cancel my appointment please",
]

print("Generating test audio using text-to-speech...")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)

audio_files = []
for i, sentence in enumerate(test_sentences):
    file_path = f"model/test_audio/sample_{i}.wav"
    tts_engine.save_to_file(sentence, file_path)
    audio_files.append((file_path, sentence))

tts_engine.runAndWait()
print("Test audio generated.\n")


# ---- Step 4: Run the pipeline on each generated audio file ----
for file_path, original_sentence in audio_files:
    audio_array, sampling_rate = sf.read(file_path)

    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    audio_array = audio_array.astype(np.float32)

    result = process_call(audio_array, sampling_rate)

    print(f"Original sentence : {original_sentence}")
    print(f"Whisper heard     : {result['transcription']}")
    print(f"Predicted intent  : {result['intent']}")
    print("---")