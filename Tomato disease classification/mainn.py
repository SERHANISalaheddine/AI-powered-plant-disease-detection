from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Specify the path to your PyTorch model (.pth file)
file_path = "C:\\Users\\salah\\OneDrive\\Bureau\\API\\model-complete_leaves.pth"

print("Model path", file_path)
MODEL = torch.load(file_path, map_location=torch.device('cpu'))  # Load the model

# Transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

CLASS_NAMES = ['Tomato___Late_blight','Tomato___healthy']
@app.get("/ping")
async def pin():
    return "Hello, I am alive"

def read_file_as_image(data) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    
    # Apply transformations to the image
    image = transform(image)
    
    # Add batch dimension
    image = image.unsqueeze(0)

    # Make predictions using the loaded PyTorch model
    with torch.no_grad():
        logits = MODEL(image)
    
    # Convert logits to probabilities using softmax
    probabilities = F.softmax(logits, dim=1)

    # Get the predicted class index
    predicted_class_index = torch.argmax(probabilities, dim=1).item()
    
    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(probabilities[0, predicted_class_index])

    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
