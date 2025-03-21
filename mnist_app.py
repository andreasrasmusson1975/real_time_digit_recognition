"""
Real-Time Digit Recognition with Streamlit and OpenCV.

This application uses OpenCV and Streamlit to capture a live webcam feed, detect handwritten 
digits in real-time, preprocess them, and classify them using a trained CNN model. 

Example
-------
>>> python -m streamlit run mnist_app.py
"""

import os
import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from mnist_cnn_classifier import MnistCnnClassifier
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import os
import streamlit as st
import time

# Get the directory of the current script
code_dir = os.path.dirname(__file__)

model_path = os.path.join(code_dir,'mnist_cnn.h5')

# Load the trained model
model = MnistCnnClassifier()
model.load_model(model_path)

# Constants
kernel = np.ones((4, 4), np.uint8)
contour_color = (0, 255, 255)
image_border = 40
# Initialize Streamlit UI
st.markdown("<h3 style='text-align: center;'>Real time handwritten digit recognition</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Andreas Rasmusson</h3>", unsafe_allow_html=True)


# Define functions
def preprocess_image(img):
    """
    Preprocesses an input image for digit recognition.

    This function converts an image to grayscale, applies binary thresholding 
    using Otsu's method, and performs morphological operations to remove noise.

    Parameters
    ----------
    img : np.ndarray
        The input image in BGR format.

    Returns
    -------
    np.ndarray
        The preprocessed binary image.
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Remove noise
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    # Thicken strokes
    img_binary = cv2.dilate(img_binary, kernel, iterations=4)
    return img_binary

def resize_and_center(img):
    """
    Resizes and centers a binary digit image to 28x28 pixels.

    The function scales the digit to fit within a 20x20 bounding box while 
    maintaining the aspect ratio. Then it places it inside a blank 28x28
    image. Finally, it centers the digit using image moments and applies 
    translation.

    Parameters
    ----------
    img : np.ndarray
        The preprocessed binary digit image.

    Returns
    -------
    np.ndarray
        A 28x28 image with the digit centered.
    """
    # Scale the digit to fit within a 20x20 bounding box
    height, width = img.shape
    scale = 20.0 / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)
    img = cv2.resize(img, (new_width, new_height))

    # Place the new 20x20 image inside a blank 28x28 image
    new_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset, y_offset = (28 - new_width) // 2, (28 - new_height) // 2
    new_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img

    # Center the image
    moments = cv2.moments(new_img)
    center_x = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 14
    center_y = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 14
    shift_x, shift_y = 14 - center_x, 14 - center_y
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered_img = cv2.warpAffine(new_img, translation_matrix, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return centered_img

class DigitRecognitionProcessor(VideoProcessorBase):
    """
    A video processing class for real-time handwritten digit recognition using OpenCV and a trained CNN model.

    This class processes each video frame from a WebRTC stream, extracts digit contours, preprocesses them, 
    resizes and centers them to 28x28 pixels, and then classifies the digit using a trained CNN model.
    The predicted digit is drawn on the video frame.

    Methods
    -------
    recv(frame)
        Processes a single video frame, detects and classifies digits, and returns the modified frame.

    Parameters
    ----------
    frame : av.VideoFrame
        A single video frame from the WebRTC stream in BGR format.

    Returns
    -------
    av.VideoFrame
        The modified video frame with detected digits outlined and labeled with predictions.
    """
    def recv(self, frame):
        """
        Processes a single video frame to detect, extract, and classify handwritten digits in real time.

        This method receives a video frame, preprocesses it, detects digit contours, extracts and resizes the digit, 
        then classifies it using a trained CNN model. The recognized digit is drawn on the frame before returning it.

        Parameters
        ----------
        frame : av.VideoFrame
            A single video frame from the WebRTC stream in BGR format.

        Returns
        -------
        av.VideoFrame
            The modified video frame with detected digits outlined and labeled with predictions.

        Processing Steps
        ----------------
        1. Converts the frame to a NumPy array.
        2. Preprocesses the image (grayscale, thresholding, noise removal).
        3. Detects contours and extracts digit regions.
        4. Ignores small or edge-bound contours to reduce false detections.
        5. Resizes and centers the digit within a 28x28 pixel frame.
        6. Classifies the digit using a CNN model.
        7. Overlays the predicted digit and bounding box on the frame.
        8. Returns the modified frame.
        """
        img = frame.to_ndarray(format='bgr24')

        # Preprocess Image
        preprocessed_img = preprocess_image(img)

        # Find contours
        contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
              # Ignore small or edge-bound contours
            if x < image_border or x + w > 1040 or y < image_border or y + h > 680 or h < 72:
                continue

            # Extract and process digit
            cropped_img = preprocessed_img[y:y + h, x:x + w]
            cropped_img = cv2.copyMakeBorder(cropped_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            centered_img = resize_and_center(cropped_img).reshape(1, -1)

            # Predict the digit
            pred = model.predict(centered_img)[0]

            # Draw bounding box and prediction
            cv2.rectangle(img, (x, y), (x + w, y + h), contour_color, 2)
            cv2.putText(img, str(pred), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, contour_color, 2)

        return frame.from_ndarray(img, format='bgr24')

# Webcam loop
ctx = webrtc_streamer(
    key='webcam',
    #mode=WebRtcMode.SENDRECV,
    video_processor_factory=DigitRecognitionProcessor,
    # media_stream_constraints={
    #     'video': {
    #         'width': {'min': 320, 'ideal': 640, 'max': 640},
    #         'height': {'min': 240, 'ideal': 480, 'max': 480},
    #         'frameRate': {'ideal': 15, 'max': 15}  # Lower frame rate for performance
    #     },
    #     'audio': False
    # },
    # frontend_rtc_configuration={
    #     'iceServers': [
    #         {'urls': ['stun:stun.l.google.com:19302']},
    #         {'urls': ['stun:stun1.l.google.com:19302']},
    #         {'urls': ['stun:global.stun.twilio.com:3478']}
    #     ]
    # },
    # server_rtc_configuration={
    #     'iceServers': [
    #         {'urls': ['stun:stun.l.google.com:19302']},
    #         {'urls': ['stun:stun1.l.google.com:19302']},
    #         {'urls': ['stun:global.stun.twilio.com:3478']}
    #     ]
    # }
)


