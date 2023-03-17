from flask import Flask, render_template, request
import face_recognition
import os

app = Flask(__name__)

# Load the known faces and their encodings
known_faces = []
known_encodings = []
name="trisha_image.jpg"
for name in os.listdir('known_faces'):
    image = face_recognition.load_image_file(os.path.join('known_faces', name))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(name.split('.')[0])
    known_encodings.append(face_encoding)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file
    file = request.files['image']

    # Save the image to a temporary directory
    filename = file.filename
    file.save(os.path.join('tmp', filename))

    # Load the image and convert it to RGB
    image = face_recognition.load_image_file(os.path.join('tmp', filename))
    rgb_image = image[:, :, ::-1]

    # Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Recognize the faces in the image
    recognized_faces = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0

        # If a match was found, use the name of the first known face
        if True in matches:
            match_index = matches.index(True)
            name = known_faces[match_index]
            confidence = 100

        recognized_faces.append({'name': name, 'confidence': confidence})

    # Return the recognized faces to the frontend
    return render_template('results.html', recognized_faces=recognized_faces)

if __name__ == '__main__':
    app.run(debug=True)
