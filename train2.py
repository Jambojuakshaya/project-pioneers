from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load data (Choose the appropriate method based on your dataset)
# Option 1: If using ImageDataGenerator (for directory-based datasets)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'path/to/train_data',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'path/to/val_data',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

# Option 2: If using custom numpy arrays (for preloaded datasets)
# X_train = np.load('train_images.npy')
# y_train = np.load('train_labels.npy')
# X_val = np.load('val_images.npy')
# y_val = np.load('val_labels.npy')

# Define your model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')  # 7 classes for emotion detection
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
    # If using custom numpy arrays, use:
    # X_train, y_train, epochs=10, validation_data=(X_val, y_val)
)

# Save the model
model.save('emotion_detection_model.h5')
