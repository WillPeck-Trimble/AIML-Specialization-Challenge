import io
import logging
import os
import urllib.request
import warnings

# Suppress noisy framework warnings — only our own print statements should show
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # Filter outTensorFlow C++ logs
os.environ["KERAS_BACKEND"] = "torch"           # must be set before importing keras
warnings.filterwarnings("ignore")               # Clean up some warning text from external libraries
logging.disable(logging.CRITICAL)               # Python logging# Silence standard python logging

import cv2
import keras_hub
import numpy as np
from PIL import Image

# 1. Load the model (Options: nano, small, medium, large, xlarge)
# 'dfine_small_coco' trades accuracy for speed and efficiency, especially without requiring powerful GPU
model = keras_hub.models.DFineObjectDetector.from_preset(
    "dfine_small_coco",
)

# COCO standard: 0=person, 15=cat, 16=dog
target_labels = {0: "Person", 15: "Cat", 16: "Dog"}

# Load an image, run inference, and print results
# Local file path  e.g. "/home/user/cat.jpg" or "cat.jpg"  python3 dfine_s_coco.py --file cat1.jpg
# HTTP/HTTPS URL   e.g. "https://example.com/photo.jpg"  python3 dfine_s_coco.py --url https://example.com/photo.jpg
# Webcam / video   pass "webcam" or a device index string like "0" to grab a single frame from the default camera, or pass a video file path to grab the first frame.  python3 dfine_s_coco.py --webcam
def _load_image(source: str) -> Image.Image:
    """Load an image, run inference, and print results

    Local file path  e.g. "/home/user/cat.jpg" or "cat.jpg"  
    example: python3 dfine_s_coco.py --file cat1.jpg

    HTTP/HTTPS URL   e.g. "https://example.com/photo.jpg"  
    example: python3 dfine_s_coco.py --url https://example.com/photo.jpg

    Webcam / video   pass "webcam" or a device index string like "0" to grab a single frame from the default camera, or pass a video file path to grab the first frame.  
    example: python3 dfine_s_coco.py --webcam
"""
    # URL
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as response:  # nosec – URL is caller-supplied
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")

    # Webcam / numeric device index
    if source.lower() == "webcam" or source.isdigit():
        device_index = 0 if source.lower() == "webcam" else int(source)
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_index}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Local video file (only grabs first frame)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    if os.path.splitext(source)[1].lower() in video_exts:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {source}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read a frame from: {source}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Local image file
    return Image.open(source).convert("RGB")


# Run the model on an Image and return filtered detections
def _run_inference(img: Image.Image, threshold: float) -> list[dict]:
    """Run the model on an Image and return filtered detections."""
    img_array = np.array(img)
    predictions = model.predict(img_array[None, ...])

    boxes = predictions["boxes"][0]
    labels = predictions["labels"][0]
    confidences = predictions["confidence"][0]

    detections = []
    for i in range(len(confidences)):
        conf = float(confidences[i])
        label_id = int(labels[i])
        if conf > threshold and label_id in target_labels:
            name = target_labels[label_id]
            print(f"Found {name}! Confidence: {conf:.2f}")
            detections.append({
                "label_id": label_id,
                "class": name,
                "confidence": conf,
                "box": boxes[i].tolist(),
            })

    if not detections:
        print("No targets detected above threshold.")
    return detections


# Identify target objects in an image from any supported source.
def identify_targets(source: str, threshold: float = 0.5) -> list[dict]:
    """
    Identify target objects in an image from any supported source.

    Parameters
    ----------
    source : str
        One of:
          - A local file path to an image (JPEG, PNG, …) or video file
          - An HTTP/HTTPS URL pointing to an image
          - "webcam" or a numeric device index string (e.g. "0") to capture
            a single live frame from a connected camera
    threshold : float
        Minimum confidence score (0–1) to report a detection.

    Returns
    -------
    list[dict]  each item has keys: label_id, class, confidence, box
    """
    img = _load_image(source)
    return _run_inference(img, threshold)

# Continuously capture frames from a webcam, run detection on each frame,
# and display an annotated live window.  Press 'q' to quit.
# WARNING: If running on Windows with WSL this requires addtional setup to access the webcam depending on your configuration
# No promises it will work in every environment, but if it does it can be a fun way to test live (I was testing with my cat and dog)
def stream_webcam(threshold: float = 0.5, device_index: int = 0) -> None:
    stream_source(device_index, threshold=threshold)


def stream_source(source, threshold: float = 0.5) -> None:
    """
    Stream detection from any OpenCV-compatible source and display an
    annotated live window.  Press 'q' to quit.

    source can be:
      - int          local camera device index (0, 1, …)
      - RTSP URL     rtsp://user:pass@192.168.1.1:554/stream
      - HLS URL      https://example.com/live/stream.m3u8
      - HTTP MJPEG   http://example.com/video.mjpg
      - local video  /path/to/video.mp4
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {source}")

    print(f"Streaming from {source} — press 'q' in the window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = _run_inference(img, threshold)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("D-FINE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="D-FINE object detector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   metavar="PATH",
                       help="Local image or video file path")
    group.add_argument("--url",    metavar="URL",
                       help="HTTP/HTTPS URL of an image")
    group.add_argument("--webcam", metavar="DEVICE", nargs="?", const="0",
                       help="Capture a single frame from a camera (default device: 0)")
    group.add_argument("--stream", metavar="DEVICE", nargs="?", const=0,
                       type=int,
                       help="Open a live annotated webcam window (press q to quit)")
    group.add_argument("--stream-url", metavar="URL",
                       help="Stream detection from an internet source: "
                            "RTSP (rtsp://…), HLS (https://….m3u8), or HTTP MJPEG (http://….mjpg)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    if args.stream_url:
        stream_source(args.stream_url, threshold=args.threshold)
    elif args.stream is not None:
        stream_webcam(threshold=args.threshold, device_index=args.stream)
    elif args.webcam is not None:
        identify_targets(args.webcam, threshold=args.threshold)
    elif args.url:
        identify_targets(args.url, threshold=args.threshold)
    else:
        identify_targets(args.file, threshold=args.threshold)