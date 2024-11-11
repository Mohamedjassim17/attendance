from flask import Flask, request, jsonify
import os
import cv2
from insightface.app import FaceAnalysis

app = Flask(__name__)

# Initialize the InsightFace model
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0)    # Use GPU if available, otherwise set ctx_id=-1 for CPU

# Define the directory for student images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_IMAGES_FOLDER = os.path.join(BASE_DIR, 'students')
TEMP_LIVE_IMAGE_PATH = os.path.join(BASE_DIR, 'temp_live_image.jpg')



# Helper function to recognize faces
def recognize_faces(live_image_path):
    recognized_students = []
    live_img = cv2.imread(live_image_path)
    if live_img is None:
        print("Failed to read the live image.")
        return recognized_students

    live_faces = face_app.get(live_img)
    if not live_faces:
        print("No faces detected in the live image.")
        return recognized_students

    print(f"Detected {len(live_faces)} face(s) in the live image.")

    for live_face in live_faces:
        live_embedding = live_face.embedding  # Get the embedding for each detected face

        for student_file in os.listdir(STUDENT_IMAGES_FOLDER):
            student_path = os.path.join(STUDENT_IMAGES_FOLDER, student_file)
            student_img = cv2.imread(student_path)

            if student_img is None:
                print(f"Failed to read student image: {student_file}")
                continue

            student_faces = face_app.get(student_img)
            if not student_faces:
                print(f"No faces detected in student image: {student_file}")
                continue

            print(f"Detected {len(student_faces)} face(s) in student image: {student_file}")
            student_embedding = student_faces[0].embedding

            # Calculate the distance between embeddings
            distance = float(cv2.norm(live_embedding - student_embedding))
            print(f"Distance between {student_file} and live image: {distance}")
        
            # Adjust this threshold based on your experimentation
            if distance < 22:  # Start with a lower value
                student_name = os.path.splitext(student_file)[0]
                recognized_students.append(student_name)

    return recognized_students



@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({'status': 'No image provided'}), 400

    try:
        # Save the received image to a temporary location
        image = request.files['image']
        image.save(TEMP_LIVE_IMAGE_PATH)
        print(f"Image saved at: {TEMP_LIVE_IMAGE_PATH}")

        # Perform face recognition
        recognized_students = recognize_faces(TEMP_LIVE_IMAGE_PATH)
        print(f"Recognized students: {recognized_students}")

        if recognized_students:
            return jsonify({
                'status': 'present',
                'students': recognized_students
            })
        else:
            return jsonify({
                'status': 'absent',
                'students': []
            })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'status': f'Error processing image: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


