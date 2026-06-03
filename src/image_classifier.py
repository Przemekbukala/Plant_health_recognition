import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from cnn_trainer_pretrained import get_cnn_model
from cnn_trainer import PlantDiseaseCNN
from diseases import DISEASE_CLASSES, parse_disease_label

NUM_CLASSES = len(DISEASE_CLASSES)
CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.20

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def _load_model(model_type, device):
    """Loads the CNN model"""
    if model_type == "pretrained":
        model = get_cnn_model(NUM_CLASSES).to(device)
        weights_path = os.path.join(MODEL_DIR, "plant_disease_best_pretrained.pth")
    else:
        model = PlantDiseaseCNN(NUM_CLASSES).to(device)
        weights_path = os.path.join(MODEL_DIR, "plant_disease_best.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def classify_image(image: Image.Image, model_type: str = "pretrained") -> dict:
    """
    Classifies a single plant leaf image. Returns prediction with confidence check.
    Recognition requires both:
      - top1 confidence >= 50%
      - margin between top1 and top2 >= 20%
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_type, device)

    img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(img_tensor)
    probabilities = F.softmax(logits, dim=1).squeeze(0)

    top5_probs, top5_indices = torch.topk(probabilities, k=min(5, NUM_CLASSES))
    top5 = [
        (DISEASE_CLASSES[idx.item()], prob.item())
        for idx, prob in zip(top5_indices, top5_probs)
    ]

    best_confidence = top5_probs[0].item()
    second_confidence = top5_probs[1].item() if len(top5_probs) > 1 else 0.0
    margin = best_confidence - second_confidence
    best_class = DISEASE_CLASSES[top5_indices[0].item()]

    rejection_reason = None
    if best_confidence < CONFIDENCE_THRESHOLD:
        rejection_reason = (
            f"Confidence too low ({best_confidence*100:.1f}% < {CONFIDENCE_THRESHOLD*100:.0f}% threshold)"
        )
    elif margin < MARGIN_THRESHOLD:
        rejection_reason = (
            f"Model is uncertain: top two predictions are too close "
            f"({best_confidence*100:.1f}% vs {second_confidence*100:.1f}%, "
            f"margin {margin*100:.1f}% < {MARGIN_THRESHOLD*100:.0f}% required)"
        )

    recognized = rejection_reason is None
    plant, disease, is_healthy = parse_disease_label(best_class) if recognized else ("", "", False)

    return {
        "predicted_class": best_class if recognized else None,
        "confidence": best_confidence,
        "margin": margin,
        "plant": plant,
        "disease": disease,
        "is_healthy": is_healthy,
        "all_predictions": top5,
        "recognized": recognized,
        "rejection_reason": rejection_reason,
    }
