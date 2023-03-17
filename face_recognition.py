import cv2
import os
import numpy as np
import sqlite3

# Set up the face recognizer
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Set up the database connection
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Create the faces table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS faces
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT,
              image BLOB)''')

# Function to insert a new face into the database
def insert_face(name, image):
    c.execute('INSERT INTO faces (name, image) VALUES (?, ?)', (name, image))
    conn.commit()

# Function to get all faces from the database
def get_faces():
    c.execute('SELECT * FROM faces')
    return c.fetchall()

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
        c.execute('SELECT name FROM faces WHERE id=?', (label,))
        name = c.fetchone()[0]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # If the 'q' key is pressed, break out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the database connection
cap.release()
cv2.destroyAllWindows()
conn.close()
