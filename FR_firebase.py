import cv2
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Set up the face recognizer
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize the Firebase app with credentials
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://<DATABASE_NAME>.firebaseio.com'
})

# Get a database reference
ref = db.reference('faces')

# Function to insert a new face into the database
def insert_face(name, image):
    ref.push({
        'name': name,
        'image': image.tolist()
    })

# Function to get all faces from the database
def get_faces():
    return ref.get()

# Function to train the face recognizer on the dataset
def train_recognizer():
    # Load the dataset
    faces = []
    labels = []
    for person in os.listdir('dataset'):
        for image_file in os.listdir(os.path.join('dataset', person)):
            image = cv2.imread(os.path.join('dataset', person, image_file), cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(int(person))

    # Train the recognizer
    recognizer.train(faces, np.array(labels))

# Train the face recognizer
train_recognizer()

# Set up the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected, recognize the person and draw a rectangle around their face
    for (x, y, w, h) in faces:
        # Extract the face region from the grayscale frame
        face_gray = gray[y:y+h, x:x+w]

        # Recognize the face using the trained model
        label, confidence = recognizer.predict(face_gray)

        # Get the name associated with the recognized label from the database
        faces_data = get_faces()
        name = faces_data[str(label)]['name']

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # If the 'q' key is pressed, break out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device
cap.release()

# Close the database connection
firebase_admin.delete_app(firebase_admin.get_app())
