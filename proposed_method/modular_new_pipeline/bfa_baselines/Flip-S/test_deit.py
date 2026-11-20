import torch
from torchvision import transforms
from PIL import Image
import timm
from transformers import AutoFeatureExtractor, DeiTForImageClassification

# Load Model
model_name = "facebook/deit-base-distilled-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = DeiTForImageClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load Test IMG
image_path = "../../00.datasets/ImageNet/val/n03759954/ILSVRC2012_val_00027379.JPEG"
img = Image.open(image_path).convert("RGB")
inputs = feature_extractor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[predicted_class_idx])
