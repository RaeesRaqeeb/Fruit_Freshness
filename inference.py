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
FRUIT_LABELS = [
    'apples',
    'banana',
    'bittergroud',
    'capsicum',
    'cucumber',
    'okra',
    'oranges',
    'potato',
    'tomato',
]
FRESH_LABELS = ['Fresh', 'Rotten']
OTHER_LABEL = os.getenv('OTHER_LABEL', 'Others')
UNKNOWN_STATUS = os.getenv('UNKNOWN_STATUS', 'Unknown')
PYTORCH_REJECT_THRESHOLD = float(os.getenv('PYTORCH_REJECT_THRESHOLD', '0.60'))
KERAS_REJECT_THRESHOLD = float(os.getenv('KERAS_REJECT_THRESHOLD', '0.60'))


# Replicate the trained architecture for inference.
class FruitInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()  # type: ignore[assignment]
        
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 9) # 9 Fruit Types
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)  # 2 Freshness states
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        return self.block2(x), self.block3(x)

# 2. Setup the Loader
def load_model():
    
    model_path = Path("model.pth")

    model = FruitInferenceModel()
    model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))
    model.eval()
    return model

# Load PyTorch model
pt_model = load_model()


def load_keras_model():
    """Load a Keras .keras saved model if available. Path can be set with MODEL_KERAS_PATH env var."""
    if tf is None:
        return None

    default_paths = [
        os.getenv('MODEL_KERAS_PATH', 'Fruit_freshness.keras'),
        'Fruit_freshness.keras',
        'model.keras',
    ]

    for p in default_paths:
        if p and Path(p).exists():
            try:
                kmodel = tf.keras.models.load_model(p)
                return kmodel
            except Exception:
                continue
    return None

# Load optional Keras model
tf_model = load_keras_model()

# Default Keras class names (18 classes) - used if no external class file found
KERAS_CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 'freshcucumber', 'freshokra',
    'freshoranges', 'freshpotato', 'freshtomato', 'rottenapples', 'rottenbanana', 'rottenbittergroud',
    'rottencapsicum', 'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

# Preprocessing matches the training pipeline.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=0, std=1),
])

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    def reject_if_needed(label: str, confidence: float, threshold: float):
        if confidence < threshold:
            return OTHER_LABEL, UNKNOWN_STATUS, True
        return label, None, False

    # PyTorch prediction
    tensor = cast(torch.Tensor, transform(img)).unsqueeze(0)
    with torch.no_grad():
        fruit_logits, fresh_logits = pt_model(tensor)
        fruit_probs = torch.softmax(fruit_logits, dim=1)[0]
        fresh_probs = torch.softmax(fresh_logits, dim=1)[0]
        fruit_idx = int(torch.argmax(fruit_probs).item())
        fresh_idx = int(torch.argmax(fresh_probs).item())
        fruit_conf = float(fruit_probs[fruit_idx].item())
        fresh_conf = float(fresh_probs[fresh_idx].item())

    fruit_label, rejected_status, fruit_rejected = reject_if_needed(
        FRUIT_LABELS[fruit_idx], fruit_conf, PYTORCH_REJECT_THRESHOLD
    )
    if fruit_rejected:
        pytorch_result = {
            "fruit_type": fruit_label,
            "status": rejected_status,
            "confidence": fruit_conf,
            "rejected": True,
        }
    else:
        pytorch_result = {
            "fruit_type": fruit_label,
            "status": FRESH_LABELS[fresh_idx],
            "confidence": fruit_conf,
            "freshness_confidence": fresh_conf,
            "rejected": False,
        }

    # Keras prediction
    keras_result = None
    if tf_model is not None:
        try:
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
                k_classes = KERAS_CLASS_NAMES

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
            keras_rejected = k0['probability'] < KERAS_REJECT_THRESHOLD
            keras_result = {
                'fruit_type': OTHER_LABEL if keras_rejected else k0['fruit_type'],
                'status': UNKNOWN_STATUS if keras_rejected else k0['status'],
                'confidence': k0['probability'],
                'rejected': keras_rejected,
                'top3': top3,
            }
        except Exception:
            keras_result = None

    def build_final_prediction(pt_result, k_result):
        pt_available = pt_result is not None
        k_available = k_result is not None

        if pt_available and k_available:
            if pt_result.get('rejected') or k_result.get('rejected'):
                return {
                    'fruit_type': OTHER_LABEL,
                    'status': UNKNOWN_STATUS,
                    'source': 'ensemble',
                    'reason': 'one_or_more_models_rejected',
                }

            if pt_result.get('fruit_type') != k_result.get('fruit_type'):
                return {
                    'fruit_type': OTHER_LABEL,
                    'status': UNKNOWN_STATUS,
                    'source': 'ensemble',
                    'reason': 'model_disagreement',
                }

            # If both models agree, use the shared fruit label.
            status = pt_result.get('status') if pt_result.get('status') in FRESH_LABELS else k_result.get('status')
            return {
                'fruit_type': pt_result.get('fruit_type'),
                'status': status or UNKNOWN_STATUS,
                'source': 'ensemble',
                'reason': 'agreement',
            }

        if pt_available:
            return {
                'fruit_type': pt_result.get('fruit_type', OTHER_LABEL),
                'status': pt_result.get('status', UNKNOWN_STATUS),
                'source': 'pytorch',
                'reason': 'keras_unavailable',
            }

        if k_available:
            return {
                'fruit_type': k_result.get('fruit_type', OTHER_LABEL),
                'status': k_result.get('status', UNKNOWN_STATUS),
                'source': 'keras',
                'reason': 'pytorch_unavailable',
            }

        return {
            'fruit_type': OTHER_LABEL,
            'status': UNKNOWN_STATUS,
            'source': 'none',
            'reason': 'no_model_available',
        }

    final_prediction = build_final_prediction(pytorch_result, keras_result)

    result = {
        # top-level keys now reflect the ensemble decision
        "fruit_type": final_prediction["fruit_type"],
        "status": final_prediction["status"],
        "decision": final_prediction,
        "pytorch": pytorch_result,
        "keras": keras_result,
        "models": {"pytorch": True, "keras": tf_model is not None},
    }

    return result