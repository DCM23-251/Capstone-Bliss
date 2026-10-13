import glob
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import cv2
import face_recognition
import numpy as np
import torch
from flask import Flask, Response, jsonify, redirect, render_template_string, request, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vision-benchmark")
torch.set_num_threads(1)
cv2.setNumThreads(1)

MODEL_PATH = Path(os.environ.get("YOLO_FIRE_MODEL", BASE_DIR / "models" / "best.pt"))
MODEL_PATH_OBJECT = Path(os.environ.get("YOLO_OBJECT_MODEL", BASE_DIR / "models" / "yolo26m.pt"))
KNOWN_FACES_DIR = Path(os.environ.get("KNOWN_FACES_DIR", BASE_DIR / "known_faces"))
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", BASE_DIR / "uploads"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

IMAGE_SIZE = int(os.environ.get("YOLO_IMAGE_SIZE", "1088"))
CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.5"))
FACE_CONFIDENCE = float(os.environ.get("FACE_RECOGNITION_CONFIDENCE", "0.6"))
FACE_SCALE_DOWN = max(1.0, float(os.environ.get("FACE_SCALE_DOWN", "2")))
TRACKER_INTERVAL = max(1, int(os.environ.get("FRAME_PROCESS_INTERVAL", "3")))
ACTIVE_CLASSES = {
    name.strip().lower()
    for name in os.environ.get("CRITICAL_CLASSES", "fire,smoke").split(",")
    if name.strip()
}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "512")) * 1024 * 1024

torch_model_lock = threading.Lock()
face_recognition_lock = threading.Lock()
pipeline_lock = threading.Lock()
primary_model = None
object_model = None
pipeline = None


def is_human_detection_name(name):
    return str(name or "").strip().lower() in {
        "person", "people", "persons", "human", "humans",
    }


def calculate_overlap_ratio(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / max(1, area_a + area_b - intersection)


def deduplicate_human_boxes(human_boxes, overlap_threshold=0.5):
    unique_boxes = []
    for detection in sorted(human_boxes, key=lambda item: item[4], reverse=True):
        if not any(
            calculate_overlap_ratio(detection[:4], existing[:4]) >= overlap_threshold
            for existing in unique_boxes
        ):
            unique_boxes.append(detection)
    return unique_boxes


def load_known_faces():
    encodings = []
    names = []
    if not KNOWN_FACES_DIR.is_dir():
        return encodings, names

    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        for filename in sorted(glob.glob(str(person_dir / "*"))):
            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                image = face_recognition.load_image_file(filename)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    names.append(person_dir.name)
                else:
                    logger.warning("No face found in reference image %s", filename)
            except (OSError, RuntimeError, ValueError):
                logger.exception("Could not load reference face image %s", filename)
    logger.info("Loaded %d known face image(s) for %d identity/identities", len(encodings), len(set(names)))
    return encodings, names


def recognize_faces_in_roi(cropped_image, bbox, known_encodings, known_names):
    if cropped_image is None or cropped_image.size == 0:
        return []

    x1, y1, _, _ = map(int, bbox)
    scale_factor = 1.0 / FACE_SCALE_DOWN
    if scale_factor != 1.0:
        cropped_image = cv2.resize(cropped_image, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    with face_recognition_lock:
        locations = face_recognition.face_locations(
            rgb_image,
            model="hog",
            number_of_times_to_upsample=1,
        )
        encodings = face_recognition.face_encodings(rgb_image, locations, num_jitters=0)

    results = []
    scale_up = 1.0 / scale_factor
    tolerance = 1.0 - FACE_CONFIDENCE
    for (top, right, bottom, left), encoding in zip(locations, encodings):
        name = "Unknown"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            if len(distances):
                best_index = int(np.argmin(distances))
                if distances[best_index] <= tolerance:
                    name = known_names[best_index]
        results.append((
            int(top * scale_up + y1),
            int(right * scale_up + x1),
            int(bottom * scale_up + y1),
            int(left * scale_up + x1),
            name,
        ))
    return results


class VisionPipeline:
    def __init__(self, source, fire_model, secondary_model):
        self.source = source
        self.fire_model = fire_model
        self.secondary_model = secondary_model
        self.known_encodings, self.known_names = load_known_faces()

        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.current_frame = None
        self.current_frame_id = 0
        self.processed_frame = None
        self.processed_frame_id = 0
        self.capture = None
        self.reader_thread = None
        self.processor_thread = None
        self.status = "Starting"
        self.last_error = None

        self.trackers = []
        self.tracker_active = False
        self.tracker_lost = False
        self.tracker_frame_count = 0
        self.face_detections = []

        self.frames_processed = 0
        self.started_at = time.monotonic()
        self.frame_times = deque(maxlen=100)
        self.processed_at = deque(maxlen=100)
        self.stage_times = {
            "yolo_primary_ms": deque(maxlen=100),
            "yolo_secondary_ms": deque(maxlen=100),
            "face_recognition_ms": deque(maxlen=100),
            "kcf_ms": deque(maxlen=100),
        }

    def start(self):
        self.reader_thread = threading.Thread(target=self._read_frames, daemon=True, name="video-reader")
        self.processor_thread = threading.Thread(target=self._process_frames, daemon=True, name="vision-processor")
        self.reader_thread.start()
        self.processor_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.capture is not None:
            self.capture.release()
        for worker in (self.reader_thread, self.processor_thread):
            if worker and worker.is_alive():
                worker.join(timeout=5)

    def reload_known_faces(self):
        self.known_encodings, self.known_names = load_known_faces()

    def _read_frames(self):
        try:
            source = self.source
            is_network_source = isinstance(source, str) and source.lower().startswith(
                ("rtsp://", "rtmp://", "http://", "https://")
            )
            is_file_source = isinstance(source, str) and not is_network_source and Path(source).is_file()
            if is_network_source:
                self.capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                self.capture = cv2.VideoCapture(source)

            if not self.capture or not self.capture.isOpened():
                raise RuntimeError(f"Could not open video source: {self.source}")

            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                if hasattr(cv2, prop):
                    self.capture.set(getattr(cv2, prop), 5000)
            if is_network_source:
                for _ in range(12):
                    if not self.capture.grab():
                        break
            file_frame_interval = 0.0
            if is_file_source:
                source_fps = self.capture.get(cv2.CAP_PROP_FPS)
                file_frame_interval = 1.0 / min(60.0, max(1.0, source_fps or 25.0))

            self.status = "Running"
            logger.info("Opened video source %s", self.source)
            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None or frame.size == 0:
                    self.status = "Source ended"
                    break
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_frame_id += 1
                if file_frame_interval and self.stop_event.wait(file_frame_interval):
                    break
        except Exception as exc:
            self.last_error = str(exc)
            self.status = "Error"
            logger.exception("Video source reader failed")
        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    def _process_frames(self):
        last_frame_id = 0
        while not self.stop_event.is_set():
            with self.frame_lock:
                frame_id = self.current_frame_id
                frame = self.current_frame.copy() if self.current_frame is not None else None
            if frame is None or frame_id == last_frame_id:
                self.stop_event.wait(0.01)
                if self.status in {"Source ended", "Error"} and frame_id == last_frame_id:
                    break
                continue
            last_frame_id = frame_id
            started = time.monotonic()
            try:
                annotated = self._process_frame(frame)
                with self.output_lock:
                    self.processed_frame = annotated
                    self.processed_frame_id += 1
                elapsed = time.monotonic() - started
                with self.metrics_lock:
                    self.frames_processed += 1
                    self.frame_times.append(elapsed)
                    self.processed_at.append(time.monotonic())
            except Exception as exc:
                self.last_error = str(exc)
                self.status = "Processing error"
                logger.exception("Vision frame processing failed")
                self.stop_event.wait(0.1)
            else:
                if self.status == "Processing error":
                    self.status = "Running"

    def _record_stage_time(self, stage, started):
        elapsed_ms = (time.monotonic() - started) * 1000
        with self.metrics_lock:
            self.stage_times[stage].append(elapsed_ms)

    def _make_kcf_tracker(self):
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF"):
            return cv2.legacy.TrackerKCF.create()
        return cv2.TrackerKCF.create()

    def _reset_trackers(self):
        self.trackers = []
        self.tracker_active = False
        self.tracker_lost = True

    def _initialize_trackers(self, frame, human_boxes):
        self.trackers = []
        for x1, y1, x2, y2, _ in human_boxes:
            width = max(1, int(x2 - x1))
            height = max(1, int(y2 - y1))
            try:
                tracker = self._make_kcf_tracker()
                if tracker.init(frame, (int(x1), int(y1), width, height)):
                    self.trackers.append(tracker)
            except (cv2.error, AttributeError, RuntimeError):
                logger.exception("Could not initialize KCF tracker")
        self.tracker_active = bool(self.trackers)
        self.tracker_lost = not self.tracker_active
        self.tracker_frame_count = 0

    def _update_trackers(self, frame):
        tracked_boxes = []
        active = []
        for tracker in self.trackers:
            try:
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, width, height = (int(value) for value in bbox)
                    tracked_boxes.append((x, y, x + width, y + height))
                    active.append(tracker)
            except cv2.error:
                logger.exception("KCF tracker update failed")
        self.trackers = active
        self.tracker_active = bool(active)
        self.tracker_lost = not self.tracker_active
        if active:
            self.tracker_frame_count += 1
        return tracked_boxes

    def _process_frame(self, frame):
        annotated = frame.copy()
        if self.tracker_active:
            started = time.monotonic()
            tracked_boxes = self._update_trackers(frame)
            self._record_stage_time("kcf_ms", started)
            if tracked_boxes and self.tracker_frame_count < TRACKER_INTERVAL:
                for x1, y1, x2, y2 in tracked_boxes:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
                self._draw_faces(annotated, self.face_detections, "Last verified: ")
                cv2.putText(
                    annotated,
                    f"Tracking Active ({len(tracked_boxes)} Persons)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )
                return annotated
            self._reset_trackers()

        detections = []
        human_boxes = []
        critical_names = set()
        self._run_yolo(self.fire_model, "yolo_primary_ms", frame, detections, human_boxes, critical_names, True)
        self._run_yolo(self.secondary_model, "yolo_secondary_ms", frame, detections, human_boxes, critical_names, False)
        human_boxes = deduplicate_human_boxes(human_boxes)

        for x1, y1, x2, y2, name, is_critical, _ in detections:
            color = (0, 0, 255) if is_critical else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                name,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        if human_boxes:
            face_started = time.monotonic()
            self.face_detections = []
            frame_height, frame_width = frame.shape[:2]
            for x1, y1, x2, y2, _ in human_boxes:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(frame_width, int(x2)), min(frame_height, int(y2))
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    self.face_detections.extend(
                        recognize_faces_in_roi(
                            roi,
                            (x1, y1, x2, y2),
                            self.known_encodings,
                            self.known_names,
                        )
                    )
            self._record_stage_time("face_recognition_ms", face_started)
            self._draw_faces(annotated, self.face_detections)
            self._initialize_trackers(frame, human_boxes)

        if critical_names:
            label = "Critical: " + ", ".join(sorted(critical_names))
            cv2.putText(annotated, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return annotated

    def _run_yolo(self, model, stage, frame, detections, human_boxes, critical_names, is_primary):
        if model is None:
            return
        started = time.monotonic()
        try:
            with torch_model_lock, torch.no_grad():
                results = model.predict(
                    frame,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    verbose=False,
                )
        finally:
            self._record_stage_time(stage, started)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                name = str(result.names.get(class_id, class_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0].item()) if box.conf is not None else CONFIDENCE
                is_critical = is_primary and name.strip().lower() in ACTIVE_CLASSES
                if is_critical:
                    critical_names.add(name)
                detections.append((x1, y1, x2, y2, name, is_critical, confidence))
                if is_human_detection_name(name):
                    human_boxes.append((x1, y1, x2, y2, confidence))

    @staticmethod
    def _draw_faces(frame, faces, prefix=""):
        for top, right, bottom, left, name in faces:
            color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame,
                f"{prefix}{name}",
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    def get_jpeg(self):
        with self.output_lock:
            if self.processed_frame is None:
                return self.processed_frame_id, None
            frame_id = self.processed_frame_id
            frame = self.processed_frame.copy()
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            raise RuntimeError("OpenCV could not encode the processed frame as JPEG")
        return frame_id, encoded.tobytes()

    def metrics(self):
        now = time.monotonic()
        with self.metrics_lock:
            average_frame_time = (
                sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
            )
            averages = {
                name: round(sum(values) / len(values), 2) if values else 0.0
                for name, values in self.stage_times.items()
            }
            observed_fps = 0.0
            if len(self.processed_at) > 1:
                elapsed = self.processed_at[-1] - self.processed_at[0]
                if elapsed > 0:
                    observed_fps = (len(self.processed_at) - 1) / elapsed
            return {
                "status": self.status,
                "last_error": self.last_error,
                "source": str(self.source),
                "frames_processed": self.frames_processed,
                "elapsed_seconds": round(now - self.started_at, 1),
                "processing_fps": round(observed_fps, 2),
                "estimated_processing_capacity_fps": (
                    round(1.0 / average_frame_time, 2) if average_frame_time else 0.0
                ),
                "average_frame_ms": round(average_frame_time * 1000, 2),
                "stage_average_ms": averages,
                "known_face_images": len(self.known_encodings),
            }


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vision Benchmark</title>
  <style>
    :root { color-scheme: dark; font: 15px system-ui, sans-serif; background: #10151d; color: #e8edf4; }
    body { max-width: 1100px; margin: 30px auto; padding: 0 18px; }
    h1 { margin-bottom: 4px; } p { color: #aab6c5; }
    form, .panel { background: #19222e; border: 1px solid #2c3b4d; border-radius: 10px; padding: 16px; margin: 16px 0; }
    input, button { font: inherit; padding: 9px; margin: 5px 4px 5px 0; }
    input[name=source] { width: min(600px, 90%); }
    button { cursor: pointer; background: #2869a6; border: 0; border-radius: 5px; color: white; }
    .layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 16px; }
    img { display: block; width: 100%; min-height: 180px; background: #080b10; border-radius: 6px; }
    dt { color: #9fb3c9; margin-top: 10px; } dd { margin-left: 0; overflow-wrap: anywhere; }
    code { color: #b8d6ff; } .error { color: #ff9b9b; }
    @media (max-width: 750px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>Computer Vision Benchmark</h1>
  <p>Dual YOLO detection, face recognition on person ROIs, and KCF tracking between detector passes.</p>
  <form action="/source" method="post" enctype="multipart/form-data">
    <label>Video URL (RTSP/RTMP/HTTP/HTTPS) or local device index:
      <input type="text" name="source" placeholder="https://example/video.mp4">
    </label><br>
    <label>Or upload a video: <input type="file" name="video" accept=".mp4,.avi,.mkv,.mov,.webm"></label>
    <button type="submit">Start / replace source</button>
  </form>
  <form action="/known-faces" method="post" enctype="multipart/form-data">
    <label>Known person's name: <input name="name" required maxlength="80"></label>
    <label>Reference photo: <input type="file" name="image" accept=".jpg,.jpeg,.png" required></label>
    <button type="submit">Add face reference</button>
    <p>Reference photos are stored in <code>known_faces/&lt;name&gt;/</code>. Face recognition still runs without references and labels detected faces Unknown.</p>
  </form>
  <div class="layout">
    <section class="panel"><h2>Processed frames</h2><img src="/video_feed" alt="Processed video stream"></section>
    <section class="panel"><h2>Live metrics</h2><dl id="metrics"></dl></section>
  </div>
  <script>
    async function refresh() {
      const response = await fetch('/status');
      const data = await response.json();
      const rows = [
        ['Status', data.status], ['Source', data.source || 'Not set'],
        ['Processed frames', data.frames_processed], ['Observed processing FPS', data.processing_fps],
        ['Estimated processing capacity', data.estimated_processing_capacity_fps + ' FPS'],
        ['Average frame time', data.average_frame_ms + ' ms'],
        ['YOLO primary average', data.stage_average_ms.yolo_primary_ms + ' ms'],
        ['YOLO secondary average', data.stage_average_ms.yolo_secondary_ms + ' ms'],
        ['Face recognition average', data.stage_average_ms.face_recognition_ms + ' ms'],
        ['KCF average', data.stage_average_ms.kcf_ms + ' ms'],
        ['Known face images', data.known_face_images]
      ];
      const metrics = document.getElementById('metrics');
      metrics.replaceChildren();
      for (const [key, value] of rows) {
        const term = document.createElement('dt');
        const detail = document.createElement('dd');
        term.textContent = key;
        detail.textContent = String(value);
        metrics.append(term, detail);
      }
      if (data.last_error) {
        const term = document.createElement('dt');
        const detail = document.createElement('dd');
        term.textContent = 'Last error';
        detail.textContent = data.last_error;
        term.className = detail.className = 'error';
        metrics.append(term, detail);
      }
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>"""


def _replace_pipeline(source):
    global pipeline
    with pipeline_lock:
        if primary_model is None or object_model is None:
            raise RuntimeError("YOLO models are not initialized")
        old_pipeline = pipeline
        pipeline = VisionPipeline(source, primary_model, object_model)
        pipeline.start()
    if old_pipeline is not None:
        old_pipeline.stop()


def _parse_source(value):
    value = (value or "").strip()
    if not value:
        raise ValueError("Enter a video URL or device index, or upload a video.")
    if value.isdecimal():
        return int(value)
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"rtsp", "rtmp", "http", "https"} or not parsed.netloc:
        raise ValueError("Video source must use RTSP, RTMP, HTTP, or HTTPS.")
    return value


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/source")
def set_source():
    uploaded_video = request.files.get("video")
    try:
        if uploaded_video and uploaded_video.filename:
            filename = secure_filename(uploaded_video.filename)
            if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
                return "Unsupported video type.", 400
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            destination = UPLOADS_DIR / f"{time.time_ns()}_{filename}"
            uploaded_video.save(destination)
            source = str(destination)
        else:
            source = _parse_source(request.form.get("source"))
        _replace_pipeline(source)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Could not start requested source")
        return str(exc), 400
    return redirect(url_for("index"))


@app.post("/known-faces")
def add_known_face():
    name = secure_filename((request.form.get("name") or "").strip())
    image = request.files.get("image")
    if not name or image is None or not image.filename:
        return "Provide a name and reference image.", 400
    filename = secure_filename(image.filename)
    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        return "Unsupported image type.", 400
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    destination = person_dir / f"{time.time_ns()}_{filename}"
    try:
        image.save(destination)
        with face_recognition_lock:
            encodings = face_recognition.face_encodings(face_recognition.load_image_file(str(destination)))
        if not encodings:
            destination.unlink(missing_ok=True)
            return "No face was found in that image.", 400
        with pipeline_lock:
            active_pipeline = pipeline
        if active_pipeline is not None:
            active_pipeline.reload_known_faces()
    except (OSError, RuntimeError, ValueError):
        logger.exception("Could not process reference face image")
        return "Could not process reference face image; see server logs.", 500
    return redirect(url_for("index"))


@app.get("/status")
def status():
    with pipeline_lock:
        active_pipeline = pipeline
    if active_pipeline is None:
        return jsonify({
            "status": "Waiting for a source",
            "last_error": None,
            "source": "",
            "frames_processed": 0,
            "elapsed_seconds": 0,
            "processing_fps": 0,
            "estimated_processing_capacity_fps": 0,
            "average_frame_ms": 0,
            "stage_average_ms": {
                "yolo_primary_ms": 0,
                "yolo_secondary_ms": 0,
                "face_recognition_ms": 0,
                "kcf_ms": 0,
            },
            "known_face_images": 0,
        })
    return jsonify(active_pipeline.metrics())


@app.get("/video_feed")
def video_feed():
    with pipeline_lock:
        active_pipeline = pipeline
    if active_pipeline is None:
        return "No video source is running.", 503

    def frames():
        last_frame_id = -1
        while not active_pipeline.stop_event.is_set():
            frame_id, jpeg = active_pipeline.get_jpeg()
            if jpeg is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            else:
                active_pipeline.stop_event.wait(0.05)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def initialize_models():
    global primary_model, object_model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Primary YOLO weights not found: {MODEL_PATH}")
    if not MODEL_PATH_OBJECT.is_file():
        raise FileNotFoundError(f"Secondary YOLO weights not found: {MODEL_PATH_OBJECT}")
    logger.info("Loading primary YOLO model: %s", MODEL_PATH)
    primary_model = YOLO(str(MODEL_PATH))
    logger.info("Loading secondary YOLO model: %s", MODEL_PATH_OBJECT)
    object_model = YOLO(str(MODEL_PATH_OBJECT))


def main():
    initialize_models()
    configured_source = os.environ.get("VIDEO_SOURCE", "").strip()
    if configured_source:
        _replace_pipeline(_parse_source(configured_source))
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
