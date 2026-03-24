import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # ✅ Correct Import

# Load the trained model
MODEL_PATH = "emotion_model.h5"  # Ensure this file exists
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Confirm input shape
print("Expected model input shape:", model.input_shape)

# ✅ Define ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "images/test",
    target_size=(48, 48),
    color_mode="grayscale",  # ✅ Force grayscale to match training
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ✅ Predict on test dataset
y_pred_probs = model.predict(test_generator)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# ✅ Get true labels
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ✅ Print results
print("\n🎯 Classification Report:\n")
from sklearn.metrics import classification_report
print(classification_report(true_labels, y_pred_labels, target_names=class_labels))
