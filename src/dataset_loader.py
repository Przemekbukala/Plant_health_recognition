import kagglehub
import os

def get_dataset(path="~/.cache/plant_dataset"):#bug - checks .cache/plant_datasets but the dataset is in .cache/kagglehub/datasets...
    """Downloads and unzips the dataset if not already present, otherwise returns the existing path."""
    #cache_dir = os.path.expanduser(path)
    #if os.path.exists(cache_dir):
    #    return cache_dir
    return kagglehub.dataset_download(
        "vipoooool/new-plant-diseases-dataset",
        #unzip=True
    )

def get_directories(path):
    """Returns the paths to the train, valid, and test directories."""
    train_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "train")
    valid_dir = os.path.join(path, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)", "valid")
    test_dir = os.path.join(path, "test", "test")
    return train_dir, valid_dir, test_dir

if __name__ == "__main__":
    dataset_path = get_dataset("~/.cache/plant_dataset")
    print("Dataset unzipped to:", dataset_path)

    train_dir, valid_dir, test_dir = get_directories(dataset_path)
    print("Train dir:", train_dir)
    print("Valid dir:", valid_dir)
    print("Test dir:", test_dir)
