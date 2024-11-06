import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter
import numpy as np
import csv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import cv2

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
torch.manual_seed(0)

# Set up paths and arguments
if __name__ == "__main__":
    dataPath = sys.argv[1]  # Path to data
    trainStatus = sys.argv[2]  # 'train' or 'test'
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "bird.pth"

# Define transformation for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((64, 64)),          # Resize images to 64x64
    transforms.RandomHorizontalFlip(),     # Data augmentation: horizontal flip
    transforms.ToTensor(),                 # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load dataset and split into train and validation sets
dataset = datasets.ImageFolder(root=dataPath, transform=transform)
class_counts = Counter([label for _, label in dataset])
total_samples = len(dataset)
num_classes = len(class_counts)

# Calculate class weights inversely proportional to class frequencies
class_weights = np.array([total_samples / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=np.float32)
class_weights_tensor = torch.tensor(class_weights).to(device)

train_size = int(0.8 * len(dataset))  # 80% of data for training
val_size = len(dataset) - train_size  # 20% for validation
train_data, val_data = random_split(dataset, [train_size, val_size])

# Data loaders for training and validation
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

# Define the CNN model
class birdClassifier(nn.Module):
    def __init__(self):
        super(birdClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize model, loss function with class weights, and optimizer
model = birdClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# CAM Visualization function
def generate_cam(model, image, target_class):
    model.eval()
    
    # Register a hook on the last convolutional layer
    final_conv = model.conv4  # Assuming conv4 is your last convolutional layer
    activations = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    hook = final_conv.register_forward_hook(forward_hook)
    
    # Forward pass
    output = model(image.unsqueeze(0).to(device))  # Add batch dimension and move to device
    hook.remove()  # Remove hook to avoid memory leak
    
    # Get the weights from the final fully connected layer for the target class
    weights = model.fc3.weight[target_class]
    
    # Generate CAM
    # activation = activations[0].squeeze().detach().cpu().numpy()
    # cam = np.zeros(activation.shape[1:], dtype=np.float32)

    # for i, w in enumerate(weights):
    #     # Move activation to CPU explicitly before using numpy
    #     activation_cpu = activation[i].cpu() if isinstance(activation[i], torch.Tensor) else activation[i]

    #     # Then use numpy() to convert it to a NumPy array
    #     cam += w * activation_cpu.numpy()
    
    # # Normalize the CAM to range [0, 1]
    # cam = np.maximum(cam, 0)
    # cam = (cam - cam.min()) / (cam.max() - cam.min())
    # cam = cv2.resize(cam, (image.shape[1], image.shape[2]))
    
    # # Overlay CAM on the original image
    # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam_image = 0.5 * heatmap + np.float32(image.permute(1, 2, 0).cpu().numpy()) / 255
    
    # return cam_image

# Training function with tracking and plotting
def train_model(num_epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), modelPath)
    print(f"Model saved to {modelPath}")

    # Plotting training and validation loss and accuracy
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_metrics.png')
    plt.show()

# Inference function
def test_model(test_dataset, model, output_csv='bird.csv'):
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results.append(predicted.item())

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])
    print(f"Predictions saved to {output_csv}")

# Run train or test based on input argument
if trainStatus == "train":
    print("Starting Training...")
    train_model()
else:
    print("Starting Inference...")
    test_data = datasets.ImageFolder(root=dataPath, transform=transform)
    test_model(test_data, model)

# Model parameter count
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params}")
