import argparse
import cv2
import rerun as rr
import rerun.blueprint as rrb
from ultralytics import YOLO


def list_video_devices():
    index = 0
    devices = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            description = cap.getBackendName()
            devices.append((index, description))
        cap.release()
        index += 1
    return devices


def capture_video(video_device_index: int, num_frames: int | None = None):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_device_index)

    if not cap.isOpened():
        raise Exception("Error: Could not open video capture.")

    frame_nr = 0

    while cap.isOpened():
        if num_frames and frame_nr >= num_frames:
            break

        # Read the frame
        ret, img = cap.read()
        if not ret:
            if frame_nr == 0:
                print("Failed to capture any frame. No camera connected?")
            else:
                print("Can't receive frame (stream end?). Exiting…")
            break

        # Get the current frame time. On some platforms it always returns zero.
        frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if frame_time_ms != 0:
            rr.set_time_nanos("frame_time", int(frame_time_ms * 1_000_000))

        rr.set_time_sequence("frame_nr", frame_nr)
        frame_nr += 1

        # Log the original image
        rr.log("image/rgb", rr.Image(img, color_model="BGR"))

        results = model(img)

        # Draw detections on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index

                if cls == 0:  # Class 0 is 'person' in COCO dataset
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"Person {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        rr.log("image/yolo", rr.Image(img, color_model="BGR"))


def main():
    parser = argparse.ArgumentParser(description="Streams a local system camera")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Which camera device to use. (Passed to `cv2.VideoCapture()`)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None, help="The number of frames to log"
    )

    rr.script_add_args(parser)

    args = parser.parse_args()

    rr.script_setup(
        args,
        "video",
        default_blueprint=rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/image/rgb", name="Video"),
            ),
            row_shares=[1, 2],
        ),
    )

    capture_video(args.device, args.num_frames)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
