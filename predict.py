import tensorflow as tf
import numpy as np
from preprocess import preprocess_image
import os

model_path = os.path.join('models', 'deepseek_model.h5')
model = tf.keras.models.load_model(model_path)

def predict_deepseek(image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)[0][0]
        return {
            "DeepSeek_Used": bool(prediction > 0.5),
            "Confidence": float(prediction)
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            "error": "Failed to process image",
            "details": str(e)
        }