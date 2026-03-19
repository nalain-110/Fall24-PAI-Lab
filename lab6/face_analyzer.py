import cv2
import dlib
import numpy as np

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(image):
    """Detect face and return 68 landmark points."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    shape = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks, face


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def measure_features(landmarks):
    """
    Extract key facial measurements from 68 landmarks.
    Landmark indices (0-based):
      0-16   Jawline
      17-21  Left eyebrow
      22-26  Right eyebrow
      27-35  Nose
      36-41  Left eye
      42-47  Right eye
      48-67  Mouth
    """
    # --- Eye measurements ---
    left_eye_width  = euclidean(landmarks[36], landmarks[39])
    right_eye_width = euclidean(landmarks[42], landmarks[45])
    avg_eye_width   = (left_eye_width + right_eye_width) / 2

    left_eye_height  = euclidean(landmarks[37], landmarks[41])
    right_eye_height = euclidean(landmarks[43], landmarks[47])
    avg_eye_height   = (left_eye_height + right_eye_height) / 2
    eye_openness     = avg_eye_height / avg_eye_width  # ratio

    # Inter-eye distance
    inter_eye_dist = euclidean(landmarks[39], landmarks[42])

    # --- Nose measurements ---
    nose_width  = euclidean(landmarks[31], landmarks[35])
    nose_length = euclidean(landmarks[27], landmarks[33])
    nose_ratio  = nose_width / nose_length

    # --- Mouth measurements ---
    mouth_width  = euclidean(landmarks[48], landmarks[54])
    mouth_height = euclidean(landmarks[51], landmarks[57])
    mouth_ratio  = mouth_height / mouth_width  # smile openness

    # --- Jaw / Face shape ---
    jaw_width   = euclidean(landmarks[0],  landmarks[16])
    face_height = euclidean(landmarks[8],  landmarks[27])  # chin to nose bridge
    face_ratio  = face_height / jaw_width   # long vs wide face

    # Cheekbone width (approx)
    cheek_width = euclidean(landmarks[2], landmarks[14])
    jaw_taper   = jaw_width / cheek_width  # 1 = square, <1 = tapered

    # --- Eyebrow arch (vertical span) ---
    left_brow_height  = abs(landmarks[19][1] - landmarks[38][1])
    right_brow_height = abs(landmarks[24][1] - landmarks[44][1])
    avg_brow_height   = (left_brow_height + right_brow_height) / 2
    brow_arch_ratio   = avg_brow_height / avg_eye_width

    return {
        "eye_openness":    round(eye_openness, 3),
        "inter_eye_dist":  round(inter_eye_dist, 1),
        "avg_eye_width":   round(avg_eye_width, 1),
        "nose_ratio":      round(nose_ratio, 3),
        "nose_length":     round(nose_length, 1),
        "mouth_ratio":     round(mouth_ratio, 3),
        "mouth_width":     round(mouth_width, 1),
        "jaw_width":       round(jaw_width, 1),
        "face_ratio":      round(face_ratio, 3),
        "jaw_taper":       round(jaw_taper, 3),
        "brow_arch_ratio": round(brow_arch_ratio, 3),
        "cheek_width":     round(cheek_width, 1),
    }


def draw_landmarks(image, landmarks, face):
    """Draw landmarks and bounding box on a copy of the image."""
    output = image.copy()

    # Bounding box
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Landmark dots
    for i, (px, py) in enumerate(landmarks):
        cv2.circle(output, (px, py), 2, (0, 120, 255), -1)

    # Region labels
    regions = {
        "Jaw":       (0, 16),
        "L.Brow":    (17, 21),
        "R.Brow":    (22, 26),
        "Nose":      (27, 35),
        "L.Eye":     (36, 41),
        "R.Eye":     (42, 47),
        "Mouth":     (48, 67),
    }
    colors = {
        "Jaw": (200, 200, 0), "L.Brow": (255, 100, 0), "R.Brow": (255, 100, 0),
        "Nose": (0, 200, 200), "L.Eye": (0, 255, 100), "R.Eye": (0, 255, 100),
        "Mouth": (200, 0, 200),
    }
    for label, (start, end) in regions.items():
        pts = landmarks[start:end + 1]
        for j in range(len(pts) - 1):
            cv2.line(output, tuple(pts[j]), tuple(pts[j + 1]), colors[label], 1)

    return output
