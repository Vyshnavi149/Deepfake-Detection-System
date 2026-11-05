
import os
import re
import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Project settings
# -----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

DATASET_DIR = "dataset"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 2. Find image files
# -----------------------------
file_paths = []
labels = []
groups = []

class_names = ["real", "fake"]

for label, class_name in enumerate(class_names):
    folder = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)

            file_paths.append(path)
            labels.append(label)

            # Keep augmented versions of the same image
            # in the same dataset split.
            group = re.sub(
                r"_aug_\d+$", "", os.path.splitext(filename)[0]
            )
            groups.append(class_name + "_" + group)

file_paths = np.array(file_paths)
labels = np.array(labels, dtype=np.int32)
groups = np.array(groups)

print("Total images:", len(file_paths))

# -----------------------------
# 3. Split dataset: 70/15/15
# -----------------------------
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)

train_idx, remaining_idx = next(
    split1.split(file_paths, labels, groups)
)

remaining_paths = file_paths[remaining_idx]
remaining_labels = labels[remaining_idx]
remaining_groups = groups[remaining_idx]

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42
)

val_idx, test_idx = next(
    split2.split(remaining_paths, remaining_labels, remaining_groups)
)

train_paths = file_paths[train_idx]
train_labels = labels[train_idx]

val_paths = remaining_paths[val_idx]
val_labels = remaining_labels[val_idx]

test_paths = remaining_paths[test_idx]
test_labels = remaining_labels[test_idx]

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))

# -----------------------------
# 4. Load and preprocess images
# -----------------------------
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def create_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_paths, train_labels, True)
val_ds = create_dataset(val_paths, val_labels)
test_ds = create_dataset(test_paths, test_labels)

# -----------------------------
# 5. Build MobileNetV2 model
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 6. Train the model
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# 7. Evaluate on test data
# -----------------------------
best_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "best_model.keras")
)

test_loss, test_accuracy = best_model.evaluate(test_ds)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# -----------------------------
# 8. Classification report
# -----------------------------
predictions = best_model.predict(test_ds)
predicted_labels = (predictions.ravel() >= 0.5).astype(int)

true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

print("\nClassification Report:")
report = classification_report(
    true_labels,
    predicted_labels,
    target_names=["Real", "Fake"],
    zero_division=0
)
print(report)

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

# -----------------------------
# 9. Save final model
# -----------------------------
best_model.save(
    os.path.join(MODEL_DIR, "deepfake_model.keras")
)

print("\nModel training and evaluation completed!")
