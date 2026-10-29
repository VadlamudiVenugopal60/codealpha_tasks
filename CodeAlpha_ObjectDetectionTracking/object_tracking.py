"""
Real-Time Object Detection and Tracking
CodeAlpha Internship - Task 4

Detects objects with a pretrained Ultralytics YOLO26 model and tracks them
across frames with ByteTrack. Works with a webcam or a video file.

Examples:
    python object_tracking.py                       # default webcam (index 0)
    python object_tracking.py --source 1            # second webcam
    python object_tracking.py --source videos/a.mp4 # video file
    python object_tracking.py --source 0 --save     # save annotated video
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    """Read command-line options so the user never has to edit the code."""
    parser = argparse.ArgumentParser(description="YOLO26 + ByteTrack object tracking")
    parser.add_argument("--source", default="0",
                        help="Webcam index (e.g. 0) or path to a video file. Default: 0")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Pretrained model name/path. 'n' (nano) is the fastest. Default: yolo26n.pt")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Minimum confidence (0-1) to keep a detection. Default: 0.4")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size. Smaller = faster, less accurate. Default: 640")
    parser.add_argument("--save", action="store_true",
                        help="Save the annotated video to output/")
    return parser.parse_args()


def open_source(source):
    """Open a webcam (if source is a number) or a video file. Exit with a clear message on failure."""
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
        if not capture.isOpened():
            sys.exit(f"[ERROR] Could not open webcam {source}. Check that it is connected, "
                     "not used by another app, and that camera permission is granted.")
    else:
        if not Path(source).is_file():
            sys.exit(f"[ERROR] Video file not found: {source}")
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            sys.exit(f"[ERROR] File exists but OpenCV cannot read it: {source}")
    return capture


def load_model(model_name):
    """Load the pretrained model. Ultralytics downloads known model names automatically on first use."""
    try:
        return YOLO(model_name)
    except Exception as error:  # download failure, corrupt file, wrong name...
        sys.exit(f"[ERROR] Could not load model '{model_name}': {error}\n"
                 "Check your internet connection (first run downloads the weights), "
                 "the model name, and that ultralytics is up to date: pip install -U ultralytics")


def create_writer(capture, frame_size, source_is_webcam):
    """Create a VideoWriter that saves to output/. frame_size is (width, height)."""
    Path("output").mkdir(exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or source_is_webcam:
        fps = 20.0  # webcams often report 0 or an unreliable value
    out_path = Path("output") / f"tracked_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        print("[WARNING] Could not create the output video; continuing without saving.")
        return None, None
    print(f"[INFO] Saving annotated video to {out_path}")
    return writer, out_path


def main():
    args = parse_args()
    # Check the video source first so a bad path/webcam fails immediately,
    # before any (possibly slow) model download.
    capture = open_source(args.source)
    is_webcam = args.source.isdigit()
    try:
        model = load_model(args.model)
    except SystemExit:
        capture.release()  # don't leave the camera/file open if the model fails
        raise

    writer = None
    smoothed_fps = 0.0
    window_name = "Object Detection and Tracking (press q to quit)"

    try:
        while True:
            start = time.time()
            ok, frame = capture.read()
            if not ok:
                print("[INFO] End of video or camera stream ended.")
                break

            # Detection + ByteTrack in one call. persist=True keeps tracker
            # state between frames, which is what keeps IDs consistent.
            results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  conf=args.conf, imgsz=args.imgsz, verbose=False)
            result = results[0]

            # plot() draws boxes, class names, confidence and track IDs.
            annotated = result.plot()

            # Objects currently detected in this frame.
            object_count = len(result.boxes)

            # FPS: smooth with a moving average so the number doesn't jitter.
            elapsed = max(time.time() - start, 1e-6)
            current_fps = 1.0 / elapsed
            smoothed_fps = current_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * current_fps

            cv2.putText(annotated, f"FPS: {smoothed_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"Objects: {object_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if args.save:
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer, _ = create_writer(capture, (width, height), is_webcam)
                    args.save = writer is not None
                if writer is not None:
                    writer.write(annotated)

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit key pressed.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # Always release resources, even if an error happened above.
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
