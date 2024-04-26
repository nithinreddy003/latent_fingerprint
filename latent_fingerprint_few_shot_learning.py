import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob  

# Set a random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Update parameters
data_dir = "E:\\latent_finger"

num_classes = 12
image_size = 508
num_shots = 5  
num_query_samples = 10  
batch_size = 2
num_epochs = 
learning_rate = 0.0001

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.samples = self.load_samples()

    def load_samples(self):
        samples = []
        for cls in self.classes:
            class_path = os.path.join(self.data_dir, cls)
            # Use glob to get a list of files
            images = glob(os.path.join(class_path, '*'))
            for img in images:
                samples.append((img, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Split dataset into train and test sets
def split_dataset(dataset, test_size=0.2):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(test_size * num_samples))
    random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, test_dataset

# Set up data transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = CustomDataset(data_dir, transform=transform)

# Split dataset into train and test sets
train_dataset, test_dataset = split_dataset(dataset, test_size=0.2)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define Prototypical Network model
class PrototypicalNet(nn.Module):
    def __init__(self, num_classes, feature_dim=512, model_name='densenet121'):
        super(PrototypicalNet, self).__init__()
        if model_name.startswith('resnet'):
            self.encoder = models.__dict__[model_name](pretrained=True)
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif model_name.startswith('densenet'):
            self.encoder = models.__dict__[model_name](pretrained=True)
            in_features = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        elif model_name.startswith('vgg'):
            self.encoder = models.__dict__[model_name](pretrained=True)
            in_features = self.encoder.classifier[6].in_features
            self.encoder.classifier = nn.Identity()
        else:
            raise NotImplementedError("Model not supported")

        self.fc = nn.Linear(in_features, feature_dim)
        self.prototypes = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the Prototypical Network model
model = PrototypicalNet(num_classes).to(device)
print(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        _, predicted_train = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Evaluation on test dataset
    model.eval()
    epoch_test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()
            _, predicted_test = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    epoch_test_loss /= len(test_loader)
    test_losses.append(epoch_test_loss)

    test_accuracy = correct_test / total_test
    test_accuracies.append(test_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

# Training & Testing Graph
plt.figure(figsize=(10, 5))

# Plot training and test losses
plt.subplot(1, 2, 1)
plt.plot(range(1, epoch + 2), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, epoch + 2), test_losses, marker='x', label='Test Loss')
plt.title('Training and Test Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and test accuracies
plt.subplot(1, 2, 2)
plt.plot(range(1, epoch + 2), train_accuracies, marker='o', label='Training Accuracy')
plt.plot(range(1, epoch + 2), test_accuracies, marker='x', label='Test Accuracy')
plt.title('Training and Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Performance Metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Set model to evaluation mode
model.eval()

# Iterate over the test dataset
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        # Forward pass
        outputs = model(images)
        # Predict class probabilities
        probabilities = nn.functional.softmax(outputs, dim=1)
        # Get predicted labels
        _, predicted = torch.max(probabilities, 1)
    # Append true and predicted labels
    true_labels.extend(labels.cpu().numpy())
    predicted_labels.extend(predicted.cpu().numpy())

# Calculate precision, recall, F1-score, and accuracy
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
accuracy = accuracy_score(true_labels, predicted_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Test Accuracy:", accuracy)
