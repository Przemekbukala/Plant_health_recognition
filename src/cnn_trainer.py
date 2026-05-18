"""
Plant Disease Classification Module

This script defines a Convolutional Neural Network (CNN) architecture and 
a training pipeline for classifying plant diseases from images using PyTorch.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from dataset_loader import get_dataset, get_directories


class PlantDiseaseCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for classifying plant diseases.
    
    Expected input: A batch of RGB images shaped as (Batch_Size, 3, 224, 224).
    Expected output: Raw prediction scores for each class.
    """

    def __init__(self, num_classes):
        """
        Initializes the CNN layers.

        Args:
            num_classes (int): The total number of plant disease categories.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch_Size, 3, 224, 224).

        Returns:
            torch.Tensor: The output predictions of shape (Batch_Size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(target_epochs):
    """
    Trains the PlantDiseaseCNN model for a specified number of epochs.

    This function handles data loading, applying transformations, initializing 
    the model, and running the training and validation loops. It handles saving 
    checkpoints and resuming training if an existing checkpoint is found.

    Args:
        target_epochs (int): The total number of epochs to train the model. 
                             If a checkpoint is found, training resumes from 
                             the last saved epoch up to the target_epochs.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset_path = get_dataset("~/.cache/plant_dataset")
    train_dir, valid_dir, test_dir = get_directories(dataset_path)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor()
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    batch_size = 64
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size*4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    model = PlantDiseaseCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_path = "plant_disease_checkpoint.pth"
    start_epoch = 0
    best_path = "plant_disease_best.pth"
    best_accuracy = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at '{checkpoint_path}'. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from Epoch {start_epoch + 1}...")
    else:
        print("No checkpoint found. Starting fresh.")

    if os.path.exists(best_path):
        best = torch.load(best_path, map_location=device, weights_only=True)
        best_accuracy = best['accuracy']

    for epoch in range(start_epoch, target_epochs):
        print(f"\nEpoch {epoch + 1}/{target_epochs}")
        model.train()
        running_loss = 0.0
        batch_count = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / batch_count
        print(f"-> Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        model.eval() 
        valid_loss = 0.0
        correct_predictions = 0
        total_images = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted_classes = torch.max(outputs.data, 1)
                total_images += labels.size(0)
                correct_predictions += (predicted_classes == labels).sum().item()

        avg_valid_loss = valid_loss / len(valid_loader)
        accuracy = 100 * correct_predictions / total_images
        print(f"-> Epoch {epoch + 1} Validation Loss: {avg_valid_loss:.4f} | Accuracy: {accuracy:.2f}%\n")

        if (epoch + 1) % 5 == 0:
            print(f"Saving checkpoint for Epoch {epoch + 1}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

        if accuracy > best_accuracy:
            print(f"Saving new best: {accuracy}...")
            best_accuracy = accuracy
            torch.save({
                'accuracy': accuracy,
                'model_state_dict': model.state_dict()
                }, best_path)

    print("\nTraining Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Plant Disease CNN")
    parser.add_argument(
        "--epochs", 
        type=int, 
        required=True, 
        help="Target number of epochs to train for."
    )
    
    args = parser.parse_args()
    train_model(target_epochs=args.epochs)
