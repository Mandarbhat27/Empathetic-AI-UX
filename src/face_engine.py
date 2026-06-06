"""
Facial Emotion Detection Engine
Uses DeepFace for webcam/image emotion detection
Runs on CPU — no GPU needed
Install: pip install deepface tf-keras opencv-python-headless
"""

import io
import numpy as np


def check_face_dependencies() -> dict:
    status = {"deepface": False, "cv2": False}
    try:
        import deepface
        status["deepface"] = True
    except ImportError:
        pass
    try:
        import cv2
        status["cv2"] = True
    except ImportError:
        pass
    return status


def detect_emotion_from_image(image_bytes: bytes) -> dict:
    """
    Detect emotion from uploaded image or webcam snapshot.
    Returns: emotion, confidence, all_scores, face_detected, error
    """
    try:
        from deepface import DeepFace
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return _error_result("Could not decode image")

        result = DeepFace.analyze(
            img,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        dominant  = result.get("dominant_emotion", "neutral").lower()
        emotions  = result.get("emotion", {})
        face_conf = result.get("face_confidence", 0)

        label_map = {
            "happy":   "happy",
            "sad":     "sad",
            "angry":   "angry",
            "fear":    "anxious",
            "disgust": "angry",
            "surprise":"excited",
            "neutral": "neutral",
        }
        mapped = label_map.get(dominant, "neutral")

        return {
            "emotion":       mapped,
            "raw_emotion":   dominant,
            "confidence":    round(face_conf * 100, 1),
            "all_scores":    {label_map.get(k, k): round(v, 1) for k, v in emotions.items()},
            "face_detected": face_conf > 0.5,
            "error":         None
        }

    except ImportError:
        return _error_result("deepface not installed — run: pip install deepface tf-keras opencv-python-headless")
    except Exception as e:
        return _error_result(str(e))


def annotate_image(image_bytes: bytes, emotion: str, confidence: float) -> bytes:
    """Draw emotion label on image and return annotated bytes"""
    try:
        import cv2

        COLOR_MAP = {
            "happy":   (80, 200, 120),
            "sad":     (180, 130, 200),
            "angry":   (80,  80, 220),
            "anxious": (80, 190, 140),
            "excited": (60, 180, 240),
            "stressed":(80, 150, 220),
            "neutral": (160, 160, 160),
        }
        LABEL_MAP = {
            "happy":   "HAPPY :)",
            "sad":     "SAD :(",
            "angry":   "ANGRY >:(",
            "anxious": "ANXIOUS :/",
            "excited": "EXCITED :D",
            "stressed":"STRESSED :|",
            "neutral": "NEUTRAL :-|"
        }

        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return image_bytes

        h, w  = img.shape[:2]
        color = COLOR_MAP.get(emotion, (160, 160, 160))
        label = f"{LABEL_MAP.get(emotion, emotion.upper())}  {confidence:.0f}%"

        cv2.rectangle(img, (0, h - 50), (w, h), (30, 30, 30), -1)
        cv2.putText(img, label, (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (3, 3), (w - 3, h - 3), color, 3)

        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    except Exception:
        return image_bytes


def _error_result(msg: str) -> dict:
    return {
        "emotion":       "neutral",
        "raw_emotion":   "neutral",
        "confidence":    0.0,
        "all_scores":    {},
        "face_detected": False,
        "error":         msg
    }