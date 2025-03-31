from notebooks.model import SkinDiseaseCNN
import torch
import torchvision.transforms as transforms
from PIL import Image


CLASS_NAMES = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinDiseaseCNN(num_classes=9).to(device)
model.load_state_dict(torch.load(r"./models/skin_disease_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_c(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    return {"class": CLASS_NAMES[predicted_class], "confidence": confidence}