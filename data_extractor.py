import os
from dataset_loader import get_dataset, get_directories
import numpy as np
from PIL import Image
import random


def image_extractor(image_dir, with_labels=True):
    """Wyciąga ścieżki do obrazów i, jeśli podane, odpowiadające im etykiety z katalogu."""
    image_paths = []
    labels = []
    if with_labels:
        for class_name in sorted(os.listdir(image_dir)):  # sorted dla przewidywalnego działania
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
    else:
        for fname in os.listdir(image_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_paths.append(os.path.join(image_dir, fname))
            labels.append(fname)  
    return image_paths, labels

def image_batch_generator(image_dir, batch_size=32, image_shape=(224,224), with_labels=True, shuffle=True):
    """
    Generator batchy obrazów z opcjonalnym shuffle.
    """
    image_paths, labels = image_extractor(image_dir, with_labels=with_labels)
    if shuffle:
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths[:], labels[:] = zip(*combined)

    total = len(image_paths)
    idx = 0
    while idx < total:
        batch_paths = image_paths[idx : idx + batch_size]
        batch_labels = labels[idx : idx + batch_size]
        batch_images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB").resize(image_shape)
            img = np.asarray(img)
            batch_images.append(img)
        yield np.array(batch_images), np.array(batch_labels)
        idx += batch_size

if __name__ == "__main__":
    dataset_path = get_dataset("~/.cache/plant_dataset")
    print("Dataset rozpakowany pod:", dataset_path)

    train_dir, valid_dir, test_dir = get_directories(dataset_path)
    print("Train dir:", train_dir)
    print("Valid dir:", valid_dir)
    print("Test dir:", test_dir)

    batch_size = 512
    image_shape = (224, 224)

    train_gen = image_batch_generator(train_dir, batch_size=batch_size, image_shape=image_shape, with_labels=True, shuffle=True)
    images, labels = next(train_gen)
    print("Train:", images.shape, labels[:10])

    valid_gen = image_batch_generator(valid_dir, batch_size=batch_size, image_shape=image_shape, with_labels= True, shuffle=True)
    images_v, labels_v = next(valid_gen)
    print("Valid:", images_v.shape, labels_v[:10])

    test_gen = image_batch_generator(test_dir, batch_size=batch_size, image_shape=image_shape, with_labels=False,shuffle=True)
    images_t, labels_t = next(test_gen)
    print("Test:", images_t.shape)

    print("Przykładowa etykieta z test:", labels_t[0])
    pil_img = Image.fromarray(images_t[0])
    pil_img.show()