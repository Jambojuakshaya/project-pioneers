import tensorflow as tf
from keras.utils import to_categorical
from keras_preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Set dataset paths
TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'

# ✅ Check if dataset folders exist
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

# 🔹 Function to load dataset into DataFrame
def createdataframe(dir):
    image_paths = []
    labels = []

    print(f"Checking directory: {dir}")  # Debugging print
    for label in os.listdir(dir):
        label_path = os.path.join(dir, label)

        # ✅ Skip non-folder files like `.ipynb_checkpoints`
        if not os.path.isdir(label_path):
            continue

        print(f"Processing label: {label}")  # Debugging print
        for imagename in os.listdir(label_path):
            image_path = os.path.join(label_path, imagename)
            image_paths.append(image_path)
            labels.append(label)

    print(f"Found {len(image_paths)} images in {dir}")  # Debugging print
    return pd.DataFrame({'image': image_paths, 'label': labels})

# 🔹 Load dataset
train = createdataframe(TRAIN_DIR)
test = createdataframe(TEST_DIR)

# ✅ Check for dataset imbalance
print("\nTraining data distribution:\n", train['label'].value_counts())

# 🔹 Function to preprocess images
def extract_features(images):
    features = []
    for image in tqdm(images):
        try:
            img = load_img(image, color_mode='grayscale', target_size=(48, 48))  # Ensure grayscale format
            img = img_to_array(img) / 255.0  # Normalize pixel values
            features.append(img)
        except Exception as e:
            print(f"❌ Error loading image {image}: {e}")
    return np.array(features)

# 🔹 Extract features
x = extract_features(train['image'])
y = train['label']

# ✅ Ensure images were loaded
if len(x) == 0:
    raise ValueError("No images were loaded! Check your dataset path.")

# 🔹 Encode labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(y), num_classes=7)

# ✅ Debugging prints
print(f"\nTotal images: {len(x)}")
print(f"Labels found: {le.classes_}")

# 🔹 Split into Train & Validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# ✅ Ensure training data exists
if len(x_train) == 0 or len(x_val) == 0:
    raise ValueError("Training or validation data is empty. Check your dataset!")

# 🔹 Define CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')  # 7 output classes
])

# 🔹 Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 🔹 Train model
history = model.fit(
    x_train, y_train, 
    validation_data=(x_val, y_val), 
    epochs=50, batch_size=32, verbose=1
)

# 🔹 Save model
model.save('emotion_model.h5')
print("\n✅ Model saved successfully as 'emotion_model.h5'!")
