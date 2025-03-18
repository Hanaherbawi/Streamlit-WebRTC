import cv2
import av
from streamlit_webrtc import webrtc_streamer

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Face detection
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="face-detection", video_frame_callback=video_frame_callback)
