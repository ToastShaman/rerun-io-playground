import argparse
import os
import cv2
import numpy as np
import zmq
from ultralytics import YOLO

YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolo11n.pt")


def main():
    parser = argparse.ArgumentParser(description="Video processing server")
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="tcp://*:5555",
        help="ZeroMQ server address to bind to (e.g., tcp://*:5555)",
    )

    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.zmq_address)

    model = YOLO(YOLO_MODEL_PATH)

    while True:
        jpeg_bytes = socket.recv()

        # Decode JPEG into an OpenCV frame
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(frame)

        # Process YOLO results
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = float(box.conf[0].item())  # Confidence score
                cls = int(box.cls[0].item())  # Class index

                detections.append(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "class": cls}
                )

        # Send results as JSON
        socket.send_json({"detections": detections})


if __name__ == "__main__":
    main()
