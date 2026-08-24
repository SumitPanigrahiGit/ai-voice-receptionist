from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch

MODEL_NAME = "openai/whisper-tiny"

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Loading sample dataset...")
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

print("Dataset loaded. Running transcription test...")

for i in range(4):
    sample = dataset[i]
    audio = sample["audio"]

    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"\nSample {i}")
    print("Predicted:", transcription)
    print("Actual:   ", sample["text"])