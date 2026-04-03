import tensorflow as tf
from src.train import build_model

CKPT_PATH = "models/ckpt.keras"
FINAL_MODEL_PATH = "models/model_v1.keras"

# Build model
model, _ = build_model()  # unpack tuple
print("✅ Model architecture created")

# Load checkpoint weights
model.load_weights(CKPT_PATH)
print("✅ Weights loaded from checkpoint")

# Save as Keras format
model.save(FINAL_MODEL_PATH)
print(f"✅ Model saved as {FINAL_MODEL_PATH}")