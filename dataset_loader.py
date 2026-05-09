import kagglehub
import os

def get_dataset(path="~/.cache/plant_dataset"):
    """Pobiera i rozpakowuje dataset, zwracając ścieżkę do katalogu z danymi."""
    cache_dir = os.path.expanduser(path)
    if os.path.exists(cache_dir):
        return cache_dir
    return kagglehub.dataset_download(
        "vipoooool/new-plant-diseases-dataset",
        unzip=True
    )

def get_directories(path):
    """Zwraca ścieżki do katalogów train, valid i test."""
    train_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "train")
    valid_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "valid")
    test_dir = os.path.join(path, "test", "test")
    return train_dir, valid_dir, test_dir

if __name__ == "__main__":
    dataset_path = get_dataset("~/.cache/plant_dataset")
    print("Dataset rozpakowany pod:", dataset_path)

    train_dir, valid_dir, test_dir = get_directories(dataset_path)
    print("Train dir:", train_dir)
    print("Valid dir:", valid_dir)
    print("Test dir:", test_dir)
