import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Veri dönüşümleri (Normalize)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Veri setini yükleme
dataset = datasets.ImageFolder('Data', transform=transform)

# Veri setinin bir alt kümesini seçme (örneğin %10)
subset_indices = np.random.choice(len(dataset), size=int(0.1 * len(dataset)), replace=False)
subset_indices = sorted(subset_indices)  # İndeksleri sıralamak hataları önleyebilir
subset = Subset(dataset, subset_indices)

# Veri setini eğitim ve doğrulama olarak bölme
train_size = int(0.8 * len(subset))
valid_size = len(subset) - train_size
train_subset, valid_subset = torch.utils.data.random_split(subset, [train_size, valid_size])

# Eğitim ve doğrulama veri yükleyicileri
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False)

# Sınıf örneklem sayıları
class_counts = [5000, 488, 67222, 13725]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

# Veri yükleyicilere ağırlıklı sampler ekleme
train_sampler_weights = [class_weights[dataset.targets[idx]] for idx in subset_indices[:train_size]]
train_sampler = WeightedRandomSampler(weights=train_sampler_weights, num_samples=len(train_sampler_weights), replacement=True)

train_loader = DataLoader(train_subset, batch_size=32, sampler=train_sampler)
valid_loader = DataLoader(valid_subset, batch_size=32, shuffle=False)

# Modelin tanımlanması
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Sınıf sayısını belirleme
num_classes = len(class_counts)
model = CNNModel(num_classes)

# Hiperparametreler
learning_rate = 0.001
epochs = 10

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Cihaz seçimi (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Eğitim, doğrulama ve grafik fonksiyonları
train_losses = []
valid_losses = []
valid_accuracies = []

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(train_loader))

        train_losses.append(running_loss/len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

def validate_model(model, valid_loader, criterion):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    valid_losses.append(valid_loss/len(valid_loader))
    valid_accuracies.append(accuracy)
    print(f"Validation Loss: {valid_loss/len(valid_loader)}, Accuracy: {accuracy}%")

def plot_metrics(train_losses, valid_losses, valid_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.show()

# Model eğitimi ve doğrulama
for epoch in range(epochs):
    train_model(model, train_loader, criterion, optimizer, epochs)
    validate_model(model, valid_loader, criterion)

# Performans grafikleri
plot_metrics(train_losses, valid_losses, valid_accuracies)

# Modeli kaydet
save_model(model, 'demans_model.pth')
print(f"Model {num_classes} sınıf ile başarıyla eğitildi ve kaydedildi.")
