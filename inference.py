import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import io
from typing import cast
import os
import requests
from pathlib import Path

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
    """Load model from disk or download from MODEL_URL if not present."""
    model_path = Path("model.pth")
    
    # If model doesn't exist, try to download from MODEL_URL env var
    if not model_path.exists():
        model_url = os.getenv("MODEL_URL")
        if model_url:
            print(f"Downloading model from {model_url}...")
            response = requests.get(model_url, timeout=300)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model saved to {model_path}")
        else:
            raise FileNotFoundError(
                f"{model_path} not found and MODEL_URL environment variable not set. "
                "Please provide MODEL_URL pointing to model.pth or ensure model.pth exists in /app."
            )
    
    model = FruitInferenceModel()
    model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Preprocessing matches the training pipeline.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=0, std=1),
])

def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = cast(torch.Tensor, transform(img)).unsqueeze(0)
    
    with torch.no_grad():
        fruit_logits, fresh_logits = model(tensor)
        fruit_idx = int(torch.argmax(fruit_logits, dim=1).item())
        fresh_idx = int(torch.argmax(fresh_logits, dim=1).item())
    
    return {
        "fruit_type": FRUIT_LABELS[fruit_idx],
        "status": FRESH_LABELS[fresh_idx],
    }
