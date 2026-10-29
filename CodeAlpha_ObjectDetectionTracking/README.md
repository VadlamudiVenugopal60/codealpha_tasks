# Real-Time Object Detection and Tracking (YOLO26 + ByteTrack + OpenCV)

CodeAlpha Internship, Task 4: Object Detection and Tracking.

A Python application that reads video from a webcam or a video file, detects objects with a pretrained Ultralytics YOLO26 model, tracks them across frames with ByteTrack, and shows bounding boxes, class labels, confidence scores, tracking IDs, an object count, and FPS.

## Features
- Webcam or video-file input (`--source`)
- Pretrained YOLO26 model, no training required
- Multi-object detection with boxes, labels and confidence
- ByteTrack tracking IDs kept across frames
- Live object count and smoothed FPS overlay
- Optional saving of annotated video to `output/` (`--save`)
- Clear errors for missing webcam, invalid video path, or model load failure
- Quit with `q`; video, writer and windows are always released

## Prerequisites
- Python 3.9 or newer (3.10+ recommended)
- A webcam (for live mode) or a video file
- Internet on the first run (model weights download automatically)

## Installation
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -U pip
pip install -r requirements.txt
```

## Usage
```bash
python object_tracking.py                        # webcam 0
python object_tracking.py --source 1             # another webcam
python object_tracking.py --source videos/test.mp4
python object_tracking.py --source 0 --save      # record annotated output
python object_tracking.py --imgsz 320 --conf 0.5 # faster / stricter
```
Press `q` in the video window to quit.

| Option | Meaning | Default |
|---|---|---|
| `--source` | Webcam index or video path | `0` |
| `--model` | Pretrained weights | `yolo26n.pt` |
| `--conf` | Minimum confidence | `0.4` |
| `--imgsz` | Inference image size | `640` |
| `--save` | Save annotated video | off |

## Project Structure
```
object_tracking.py   main application
requirements.txt     dependencies
README.md            documentation
.gitignore           keeps weights and videos out of Git
screenshots/         your real screenshots (add them yourself)
output/              saved videos (not committed)
```

## Example Output
_Add your own screenshots here after running the app:_
`![Webcam detection](screenshots/webcam_detection.png)`

The window shows a coloured box per object, labelled with an ID, class name and confidence (e.g. `id:3 person 0.87`), plus FPS and object count in the top-left corner.

## Limitations
- Speed depends on your CPU/GPU, model size and image size; no fixed FPS is guaranteed.
- IDs can switch or be lost during long occlusions, fast motion, or when objects look alike.
- Detects only the 80 COCO classes the pretrained model knows.
- Low light and motion blur reduce accuracy.
- Count shows objects in the current frame, not unique objects seen overall.

## Tested On
_Fill in honestly after your testing: OS, Python version, package versions, hardware._
