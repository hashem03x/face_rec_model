from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import pickle
import io
from PIL import Image

app = Flask(__name__)

# === Load model and label encoder ===
model = load_model('face_classifier_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# === Initialize FaceNet & MTCNN ===
embedder = FaceNet()
detector = MTCNN()

# === Face Preprocessing ===
def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (160, 160))
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

# === Get Face Embedding ===
def get_face_embedding(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]['box']
    face_img = rgb_frame[y:y+h, x:x+w]
    preprocessed = preprocess_face(face_img)
    embedding = embedder.embeddings(preprocessed)
    
    return embedding[0], (x, y, w, h)

@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    embedding, face_box = get_face_embedding(frame)
    if embedding is None:
        return jsonify({'error': 'No face detected'}), 400

    prediction = model.predict(np.expand_dims(embedding, axis=0))[0]
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = float(np.max(prediction))

    return jsonify({
        'label': predicted_label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
