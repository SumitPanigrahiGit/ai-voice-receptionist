from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Using whisper-tiny since we're on CPU locally (no free GPU like Colab had)
MODEL_NAME = "openai/whisper-tiny"

print("Loading Whisper processor and model...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Model loaded successfully.")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")