import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
checkpoint = torch.load("saved_model_epoch_8.pt")

model.load_state_dict(checkpoint['model_state_dict'])

image = preprocess(Image.open("/home/arthur/Downloads/KITTI_DATASET_ROOT/training/image_2/000001.png")).unsqueeze(0).to(device)
text = clip.tokenize(["There are 1 car and 1 truck in the image.", "There is 1 car in the image.", "There is 1 truck in the image."]).to(device)

model.eval()
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)