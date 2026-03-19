import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from face_analyzer import get_landmarks, measure_features, draw_landmarks
from personality import classify_mbti, MBTI_PROFILES

app = Flask(__name__)
RESULT_FOLDER = os.path.join("static", "result")
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Resize for consistent processing
    h, w = image.shape[:2]
    if w > 800:
        scale = 800 / w
        image = cv2.resize(image, (800, int(h * scale)))

    # Detect landmarks
    landmarks, face = get_landmarks(image)
    if landmarks is None:
        return jsonify({"error": "No face detected. Please use a clear front-facing photo."}), 400

    # Measure features
    measurements = measure_features(landmarks)

    # Classify personality
    mbti_code, scores = classify_mbti(measurements)
    profile = MBTI_PROFILES.get(mbti_code, {})

    # Draw and save annotated image
    annotated = draw_landmarks(image, landmarks, face)
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(save_path, annotated)

    return jsonify({
        "mbti": mbti_code,
        "profile": profile,
        "measurements": measurements,
        "scores": scores,
        "image_url": f"/static/result/{filename}",
    })


@app.route("/static/result/<filename>")
def result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
