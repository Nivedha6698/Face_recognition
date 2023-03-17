from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

cred = credentials.Certificate("path/to/firebase/credentials.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
# Function to get all images from the database
def get_images():
    images = []
    docs = db.collection('images').stream()
    for doc in docs:
        image = np.frombuffer(doc.to_dict()['image'], dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        images.append(image)
    return images

# Function to train the face recognition model on the dataset
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = get_images()
    faces = []
    labels = []
    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            faces.append(face_gray)
            labels.append(i)
    recognizer.train(faces, np.array(labels))
    return recognizer

# Function to store a new image in the database
def store_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Code to process the uploaded image
    image = cv2.imread(request.files['image'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return redirect(url_for('result'))

@app.route('/result')
def result():
    # Code to display the result
    return render_template('result.html')
