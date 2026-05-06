import os
import io
from pathlib import Path
from typing import cast

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

# Optional TensorFlow/Keras support for side-by-side inference
try:
    import tensorflow as tf  # type: ignore[import-not-found]
except Exception:
    tf = None

# The label order matches the LabelEncoder order used during training.

# FRESH_LABELS = ['Fresh', 'Rotten']
OTHER_LABEL = os.getenv('OTHER_LABEL', 'Others')
UNKNOWN_STATUS = os.getenv('UNKNOWN_STATUS', 'Unknown')
KERAS_REJECT_THRESHOLD = float(os.getenv('KERAS_REJECT_THRESHOLD', '0.60'))



def load_keras_model():
    """Load a Keras .keras saved model if available. Path can be set with MODEL_KERAS_PATH env var."""
    if tf is None:
        return None
    print("TensorFlow/Keras is available. Attempting to load Keras model...")
    
    try:
        kmodel = tf.keras.models.load_model('Model_1.keras')
        return kmodel
    except Exception as e:
        print(f"Error loading Keras model: {e}")
    return None

# Load optional Keras model
tf_model = load_keras_model()

# Default Keras class names (18 classes) - used if no external class file found
KERAS_CLASS_NAMES_model_1 = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 'freshcucumber', 'freshokra',
    'freshoranges', 'freshpotato', 'freshtomato', 'rottenapples', 'rottenbanana', 'rottenbittergroud',
    'rottencapsicum', 'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]
KERAS_CLASS_NAMES_model_2 =['Apple_Bad', 'Apple_Good', 'Banana_Bad', 'Banana_Good', 'Guava_Bad', 'Guava_Good', 'Lime_Bad', 'Lime_Good', 'Orange_Bad', 'Orange_Good']

# Preprocessing matches the training pipeline.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize(mean=0, std=1),
])

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    def reject_if_needed(label: str, confidence: float, threshold: float):
        if confidence < threshold:
            return OTHER_LABEL, UNKNOWN_STATUS, True
        return label, None, False

    
    # Keras prediction
    keras_result = None
    print("Running Keras model inference...")
    if tf_model is not None:
        try:
            print("Preprocessing image for Keras model...")
            kimg = img.resize((128, 128))
            karr = np.asarray(kimg).astype(np.float32) / 255.0
            if karr.ndim == 2:
                karr = np.stack([karr, karr, karr], axis=-1)
            karr = np.expand_dims(karr, axis=0)
            preds = tf_model.predict(karr)
            k_probs = np.asarray(preds[0])
            # top-3 indices
            topk = list(np.argsort(k_probs)[::-1][:3])

            # Load classes
            k_names_path = Path(os.getenv('MODEL_KERAS_CLASSES', 'keras_class_names.json'))
            k_classes = None
            if k_names_path.exists():
                try:
                    import json as _json
                    raw = _json.loads(k_names_path.read_text())
                    if isinstance(raw, list) and len(raw) >= 1:
                        k_classes = raw
                except Exception:
                    k_classes = None
            if k_classes is None:
                k_classes = KERAS_CLASS_NAMES_model_1

            def split_label(name: str):
                
                if name.startswith('fresh'):
                    return name[len('fresh'):], 'Fresh'
                if name.startswith('rotten'):
                    return name[len('rotten'):], 'Rotten'
                
                for p in ['fresh', 'rotten']:
                    if p in name:
                        rest = name.replace(p, '')
                        return rest, p.capitalize()
                return name, 'Unknown'

            top3 = []
            for idx in topk:
                label = k_classes[idx] if idx < len(k_classes) else f'class_{idx}'
                prob = float(k_probs[idx])
                fruit_name, status_name = split_label(label)
                top3.append({
                    'label': label,
                    'probability': prob,
                    'fruit_type': fruit_name,
                    'status': status_name,
                })

            # primary Keras pick (top-1)
            k0 = top3[0]
            keras_rejected = k0['probability'] < 0.85
            keras_result = {
                'fruit_type': OTHER_LABEL if keras_rejected else k0['fruit_type'],
                'status': UNKNOWN_STATUS if keras_rejected else k0['status'],
                'confidence': k0['probability'],
                'rejected': keras_rejected,
                'top3': top3,
            }
        except Exception as e:
            print(f"Error during Keras inference: {e}")
            keras_result = None

    def build_final_prediction( k_result):
        # pt_available = pt_result is not None
        k_available = k_result is not None

        if k_available:
            if k_result.get('rejected'):
                return {
                    'fruit_type': OTHER_LABEL,
                    'status': UNKNOWN_STATUS,
                    'reason': 'model_rejected',
                }


        if k_available:
            return {
                'fruit_type': k_result.get('fruit_type', OTHER_LABEL),
                'status': k_result.get('status', UNKNOWN_STATUS),
                'source': 'keras',
            }

        return {
            'fruit_type': OTHER_LABEL,
            'status': UNKNOWN_STATUS,
            'source': 'none',
            'reason': 'no_model_available',
        }

    final_prediction = build_final_prediction(keras_result)

    result = {
        # top-level keys now reflect the ensemble decision
        "fruit_type": final_prediction["fruit_type"],
        "status": final_prediction["status"],
        "decision": final_prediction,
        "keras": keras_result,
        "models": {"keras": tf_model is not None},
    }

    return result