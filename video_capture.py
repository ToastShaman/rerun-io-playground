import argparse
import cv2
import zmq
import rerun as rr


def capture_video(video_device_index, num_frames, callback):
    """
    Capture video from a camera and process each frame with a callback function.
    """
    try:
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
    finally:
        cap.release()


def callback_factory(zmq_address):
    """
    Create a callback function that processes each frame from the camera.
    """

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

        # Send the image to the ZeroMQ server for processing
        zmq_socket.send(jpeg_frame.tobytes())

        # Wait for the reply
        data = zmq_socket.recv_json()

        # Draw bounding boxes
        for r in data["tracking"]:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            conf = r["conf"]
            cls = r["class"]
            track_id = r["track_id"]

            if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display label
                label = f"ID {track_id}: {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
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

    return zmq_socket, context, callback


def main(device_idx, num_frames, zmq_address):
    zmq_socket, context, callback = callback_factory(zmq_address)

    try:
        capture_video(device_idx, num_frames, callback)
    finally:
        zmq_socket.close()
        context.term()


if __name__ == "__main__":
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

    args = parser.parse_args()

    rr.init("retail-analytics-demo", spawn=True)

    main(args.device, args.num_frames, args.zmq_address)
