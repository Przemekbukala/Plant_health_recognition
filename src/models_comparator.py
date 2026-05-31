import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from dataset_loader import get_dataset, get_directories
from sklearn.metrics import confusion_matrix
import numpy as np

from cnn_trainer_pretrained import get_cnn_model
from cnn_trainer import PlantDiseaseCNN

def get_cnn_model_custom(num_classes, device):
    model = PlantDiseaseCNN(num_classes).to(device)
    if os.path.exists("plant_disease_best.pth"):
        best = torch.load("plant_disease_best.pth", map_location=device, weights_only=True)
        model.load_state_dict(best['model_state_dict'])
    else:
        raise Exception("Brak zapisanego modelu customowego!")
    return model

def get_cnn_model_pretrained(num_classes, device):
    model = get_cnn_model(num_classes).to(device)
    if os.path.exists("plant_disease_best_pretrained.pth"):
        best = torch.load("plant_disease_best_pretrained.pth", map_location=device, weights_only=True)
        model.load_state_dict(best['model_state_dict'])
    else:
        raise Exception("Brak zapisanego modelu pretrained!")
    return model


def compare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testowanie na {device}...")

    dataset_path = get_dataset("~/.cache/plant_dataset")
    train_dir, valid_dir, _ = get_directories(dataset_path)
    #test_dir = "~/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2/test_fixed"
    test_dir = valid_dir

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    train_dataset = datasets.ImageFolder(root=train_dir)
    total_classes = len(train_dataset.classes)
    
    idx_to_class_train = {v: k for k, v in train_dataset.class_to_idx.items()}

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"Liczba obrazków testowych: {len(test_dataset)}")
    
    model_custom = get_cnn_model_custom(total_classes, device)
    model_pretrained = get_cnn_model_pretrained(total_classes, device)
    
    model_custom.eval()
    model_pretrained.eval()
    
    correct_custom = 0
    correct_pretrained = 0
    total = 0

    with torch.no_grad():
        y_pred_custom = []
        y_pred_pretrained = []
        y_true = []
        for images, labels in test_loader:
            images = images.to(device)
            
            true_disease_names = [test_dataset.classes[label] for label in labels]
            
            out_custom = model_custom(images)
            out_pretrained = model_pretrained(images)
            
            _, pred_idx_custom = torch.max(out_custom, 1)
            _, pred_idx_pretrained = torch.max(out_pretrained, 1)

            for i in range(len(images)):
                model_guess_custom = idx_to_class_train[pred_idx_custom[i].item()]
                model_guess_pretrained = idx_to_class_train[pred_idx_pretrained[i].item()]
                true_answer = true_disease_names[i]

                if model_guess_custom == true_answer:
                    correct_custom += 1
                if model_guess_pretrained == true_answer:
                    correct_pretrained += 1
                
                total += 1

            y_pred_custom.extend(pred_idx_custom.cpu().tolist())
            y_pred_pretrained.extend(pred_idx_pretrained.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
            

    acc_custom = 100 * correct_custom / total
    acc_pretrained = 100 * correct_pretrained / total

    print(f"\n--- WYNIKI ---")
    print(f"CUSTOM:      {acc_custom:.2f}% poprawnych ({correct_custom}/{total})")
    print(f"PRETRAINED:  {acc_pretrained:.2f}% poprawnych ({correct_pretrained}/{total})")

    conf_matrix_custom = confusion_matrix(y_true, y_pred_custom)
    conf_matrix_pretrained = confusion_matrix(y_true, y_pred_pretrained)

    print("\n\n\n--- MACIERZE POMYŁEK --\n")
    print("CUSTOM:")
    print(conf_matrix_custom)
    print("\n\nPRETRAINED:")
    print(conf_matrix_pretrained)

    with open("./docs/conf_matrix.txt", "w") as f:
        f.write("CUSTOM:")
        f.write(np.array2string(conf_matrix_custom, threshold=np.inf))
        f.write("\n\nPRETRAINED:")
        f.write(np.array2string(conf_matrix_pretrained, threshold=np.inf))




if __name__ == "__main__":
    compare_models()