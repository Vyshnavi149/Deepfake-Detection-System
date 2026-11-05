
import cv2
import numpy as np
import os

IMG_SIZE = 128

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print("Unable to read:", image_path)
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values between 0 and 1
    image = image.astype("float32") / 255.0

    return image


def load_dataset():
    images = []
    labels = []

    folders = [
        ("dataset/real", 0),
        ("dataset/fake", 1)
    ]

    for folder, label in folders:
        if not os.path.exists(folder):
            print("Folder not found:", folder)
            continue

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)

            image = preprocess_image(path)

            if image is not None:
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    X, y = load_dataset()

    print("Dataset loading completed!")
    print("Total images:", len(X))
    print("Real images:", np.sum(y == 0))
    print("Fake images:", np.sum(y == 1))
    print("Image shape:", X.shape)
