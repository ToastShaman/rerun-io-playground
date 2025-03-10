import argparse
import json
import os
import cv2
import numpy as np
import zmq
from ultralytics import YOLO

YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolo11n.pt")

YOLO_TRACKER_PATH = os.path.join(os.path.dirname(__file__), "models", "bytetrack.yaml")


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
        detections = model(frame)

        # Process YOLO results to extract bounding boxes and information
        detections_data = []
        for det in detections:
            for box in det.boxes:
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract the confidence score and class index
                conf = float(box.conf[0].item())
                cls = int(box.cls[0].item())

                response = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf,
                    "class": cls,
                }

                detections_data.append(response)

        # Run YOLO tracker
        tracked_objects = model.track(frame, persist=True, tracker=YOLO_TRACKER_PATH)

        # Process YOLO results to extract bounding boxes and information
        tracking_data = []
        for obj in tracked_objects:
            for box in obj.boxes:
                # Extract the track ID
                track_id = int(box.id[0].item())

                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract the confidence score and class index
                conf = float(box.conf[0].item())
                cls = int(box.cls[0].item())

                response = {
                    "track_id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf,
                    "class": cls,
                }

                tracking_data.append(response)

        # Send results as JSON
        responses = {"detections": detections_data, "tracking": tracking_data}

        print(json.dumps(responses))

        socket.send_json(responses)


if __name__ == "__main__":
    main()
