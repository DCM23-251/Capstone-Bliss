# Standalone vision benchmark

This directory is an isolated Flask benchmark for the same core vision flow used by the main app:

1. Run the fire/smoke YOLO model and the secondary YOLO26m model with `imgsz=1088`.
2. Merge overlapping person boxes from both models using the main app's IoU threshold of `0.5`.
3. Run HOG face detection and face matching on person ROIs resized to one-half scale.
4. Initialize KCF trackers for detected people and use them for up to two frames before the next full YOLO/face pass.
5. Render the detections, recognized/unknown faces, and KCF boxes in the Flask MJPEG view.

The Docker build downloads the same two model files from the pinned Hugging Face revisions used by the project's existing Dockerfile. No project Python modules, templates, or model directory are copied into this image.

## Run locally

From this directory:

```powershell
python -m pip install -r requirements.txt
python fork_main.py
```

The model weights must be available at `models/best.pt` and `models/yolo26m.pt` for a local run. Override their paths with `YOLO_FIRE_MODEL` and `YOLO_OBJECT_MODEL`.

Open `http://localhost:5000`, then enter an RTSP/RTMP/HTTP(S) stream URL in the **Video URL** form and click **Start stream**. Video uploads use a separate form so an accidentally selected, large video is not sent with an RTSP URL. Alternatively, set `VIDEO_SOURCE` before starting the app; a numeric value selects a local camera device. Add face-reference photos in the page, or mount them under `known_faces/<person-name>/`.

## Railway / Docker

Use `vision_benchmark` as the Railway service root so its `Dockerfile` and `requirements.txt` are used. Set `VIDEO_SOURCE` to a reachable camera/video URL to begin processing automatically, or leave it unset and submit a URL/upload in the web UI. Railway should provide `PORT`.

Video uploads are limited by `MAX_UPLOAD_MB` (default `512`), and the hosting platform or reverse proxy may impose a lower request-size limit. For camera testing, enter the stream link in the URL form rather than using the upload form.

Useful environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `YOLO_IMAGE_SIZE` | `1088` | YOLO inference image size |
| `DETECTION_CONFIDENCE` | `0.5` | Confidence threshold for both YOLO models |
| `FACE_RECOGNITION_CONFIDENCE` | `0.6` | Face-match confidence used by the main app |
| `FACE_SCALE_DOWN` | `2` | Person ROI downscale factor before face recognition |
| `FRAME_PROCESS_INTERVAL` | `3` | YOLO/face interval, with KCF used between detector passes |
| `CRITICAL_CLASSES` | `fire,smoke` | Primary-model classes highlighted as critical |
| `KNOWN_FACES_DIR` | `known_faces` | Directory for named reference-face subfolders |
| `MAX_UPLOAD_MB` | `512` | Maximum uploaded video size |

The status endpoint and page report recent observed output FPS and a rolling window of up to 100 processed frames. Frame latency averages include both full detector frames and KCF-only frames; the UI reports their counts and separate average latencies. Each stage average is per invocation within that same output-frame window, with its sample count shown, so stage averages should not be summed as if every stage ran on every frame. Estimated processing capacity is the reciprocal of mean processing time per output frame; observed FPS is based on wall-clock intervals between completed frames and can be lower due to source/network stalls. These are process-level benchmark measurements, not Railway billing/credit estimates.

This minimal test UI has no authentication. Use it only in a private test deployment and avoid exposing camera feeds or face-reference images publicly. Uploaded videos and reference images are stored on the container filesystem.
