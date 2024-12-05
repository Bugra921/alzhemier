import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO

# Model tanımı (CNN Model)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modeli yükle
model = CNNModel(num_classes=4)
model.load_state_dict(torch.load('demans_model.pth', map_location=torch.device('cpu')))
model.eval()

# Sınıf isimleri
class_names =['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
# Görüntü ön işleme fonksiyonu
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = transform(img).unsqueeze(0)  # Batch boyutuna dönüştür
    return img

# Tahmin yapma fonksiyonu
def predict_image(img):
    img = preprocess_image(img)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

# Streamlit arayüzü
st.title("Demans Sınıflandırma Uygulaması")

camera_input = st.camera_input('Kameradan resim çek')
gallery_input = st.file_uploader('VEYA Tomografi Resmi Yükleyin', accept_multiple_files=False)

if camera_input is not None:
    img_bytes = camera_input.getvalue()
    img = Image.open(BytesIO(img_bytes)).convert('L')  # Görüntüyü siyah-beyaza dönüştür

    predicted_class, confidence = predict_image(img)
    st.write(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

elif gallery_input is not None:
    img_bytes = gallery_input.getvalue()
    img = Image.open(BytesIO(img_bytes)).convert('L')  # Görüntüyü siyah-beyaza dönüştür

    predicted_class, confidence = predict_image(img)
    st.write(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
    st.write(f"İnanılırlık Yüzdesi: {confidence*100:.2f}%")

else:
    st.write("Lütfen bir resim yükleyin veya kamera kullanarak bir resim çekin.")
