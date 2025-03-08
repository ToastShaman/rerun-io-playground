import argparse
import json
import cv2
import rerun as rr
import zmq
import rerun.blueprint as rrb


def capture_video(video_device_index, num_frames, callback):
    cap = cv2.VideoCapture(video_device_index)

    if not cap.isOpened():
        raise Exception("Error: Could not open video capture.")

    frame_nr = 0

    while cap.isOpened():
        if num_frames and frame_nr >= num_frames:
            break

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to receive frame from video capture.")

        # Get the current frame time. On some platforms it always returns zero.
        frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # process the image
        callback(frame_time_ms, frame_nr, frame)

        frame_nr += 1


def callback_factory(zmq_address):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.REQ)
    zmq_socket.connect(zmq_address)

    def callback(frame_time_ms, frame_nr, frame):
        if frame_time_ms != 0:
            rr.set_time_nanos("frame_time", int(frame_time_ms * 1_000_000))

        rr.set_time_sequence("frame_nr", frame_nr)

        ret, jpeg_frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            raise Exception("Failed to encode frame as JPEG")

        # Log the original image
        rr.log(
            "image/original",
            rr.EncodedImage(contents=jpeg_frame, media_type="image/jpeg"),
        )

        # Send the image to the ZeroMQ server
        zmq_socket.send(jpeg_frame.tobytes())

        # Wait for the reply
        message = zmq_socket.recv()

        # Draw bounding boxes
        detections = json.loads(message)
        for det in detections["detections"]:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["conf"]
            cls = det["class"]

            if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display label (confidence + class)
                label = f"Class {cls}: {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Encode frame with boxes as JPEG
        ret, jpeg_frame_with_boxes = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
        )
        if not ret:
            raise Exception("Failed to encode processed frame as JPEG")

        # Log the processed image with bounding boxes
        rr.log(
            "image/yolo",
            rr.EncodedImage(contents=jpeg_frame_with_boxes, media_type="image/jpeg"),
        )

    return callback


def main():
    parser = argparse.ArgumentParser(description="Capture video from a camera")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Which camera device to use. (Passed to `cv2.VideoCapture()`)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None, help="The number of frames to log"
    )
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="tcp://localhost:5555",
        help="ZeroMQ server address (e.g., tcp://localhost:5555)",
    )

    rr.script_add_args(parser)

    args = parser.parse_args()

    rr.init("retail-analytics-demo", spawn=True)

    rr.script_setup(
        args,
        "retail-analytics-demo",
        default_blueprint=rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/image/original", name="Video"),
                rrb.Spatial2DView(origin="/image/yolo", name="YOLO Detector"),
            ),
        ),
    )

    callback = callback_factory(args.zmq_address)

    capture_video(args.device, args.num_frames, callback)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
