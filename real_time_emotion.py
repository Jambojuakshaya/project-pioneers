import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Load the trained model
MODEL_PATH = "emotion_model.h5"
model = load_model(MODEL_PATH)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Load OpenCV’s pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to match model input
        face = img_to_array(face) / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict emotion
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        # Display emotion on video frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show the video feed
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
