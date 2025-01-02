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
@st.cache_resource
def load_model(model_path, num_classes):
    model = CNNModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('demans_model.pth', num_classes=4)

# Sınıf isimleri
class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']

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

uploaded_files = st.file_uploader('Tomografi Resimlerini Yükleyin (en fazla 3 adet)', accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    predictions = []
    confidences = []

    for file in uploaded_files[:3]:  # Maksimum 3 dosya işleme
        img_bytes = file.getvalue()
        img = Image.open(BytesIO(img_bytes)).convert('L')  # Görüntüyü siyah-beyaza dönüştür
        predicted_class, confidence = predict_image(img)
        predictions.append(predicted_class)
        confidences.append(confidence)
        st.write(f"Yüklenen Resim: {file.name}")
        st.write(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
        st.write(f"İnanılırlık Yüzdesi: {confidence * 100:.2f}%")
    
    # Ortalama tahmin ve güven
    avg_prediction = np.mean(predictions)
    avg_confidence = np.mean(confidences)
    st.write(f"\nOrtalama Tahmin Edilen Sınıf: {class_names[int(avg_prediction)]}")
    st.write(f"Ortalama İnanılırlık Yüzdesi: {avg_confidence * 100:.2f}%")

else:
    st.write("Lütfen en fazla 3 resim yükleyin.")
