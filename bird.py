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

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
torch.manual_seed(0)

# Hook to store inputs and outputs
#class Hook:
    #def __init__(self, layer):
        #self.layer = layer
        #self.inputs = None
        #self.outputs = None
        #self.hook = layer.register_forward_hook(self.forward_hook)

    #def forward_hook(self, module, input, output):
        #self.inputs = input
        #self.outputs = output

    #def close(self):
        #self.hook.remove()

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

# Count the number of samples in each class
class_counts = Counter([label for _, label in dataset])
print("Class counts:", class_counts)  # Debugging line
total_samples = len(dataset)
num_classes = len(class_counts)

# Calculate class weights inversely proportional to class frequencies
class_weights = np.array([total_samples / (num_classes * class_counts[i]) for i in range(num_classes)], dtype=np.float32)

# Ensure class_weights is a tensor of shape [num_classes]
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Increased filters
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Increased filters
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Increased filters
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Increased filters
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Increased output size
        self.fc2 = nn.Linear(1024, 512)  # Increased output size
        self.fc3 = nn.Linear(512, 10)  # Assuming 10 bird classes
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

        # Create hooks to track inputs and outputs
        #self.hooks = []
        #for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2, self.fc3]:
            #self.hooks.append(Hook(layer))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


    #def get_hook_data(self):
        #return [(hook.layer.__class__.__name__, hook.inputs, hook.outputs) for hook in self.hooks]

    #def close_hooks(self):
        #for hook in self.hooks:
            #hook.close()

# Initialize model, loss function with class weights, and optimizer
model = birdClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training function
def train_model(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)  # Move data to device
            labels = labels.to(device)
            #print(f"Images device: {images.device}, Labels device: {labels.device}")  # Debugging
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), modelPath)
    print(f"Model saved to {modelPath}")

# Inference function
def test_model(test_dataset, model, output_csv='bird.csv'):
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Set shuffle to False
    results = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)  # Move data to device
            #labels = labels.to(device)  # Ensure labels are on the correct device
            #print(f"Images device: {images.device}, Labels device: {labels.device}")  # Debugging
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results.append(predicted.item())

    # Write only predicted labels to a CSV file
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

# After testing, you can print the hook data for inspection
#hook_data = model.get_hook_data()
#for layer_name, inputs, outputs in hook_data:
#    print(f"Layer: {layer_name}\nInputs: {inputs}\nOutputs: {outputs}\n")

# Model parameter count
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params}")
