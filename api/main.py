from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.inference_pipeline import PipelineConfig, UnifiedFaceSecurityPipeline
from api.settings import Settings
from utils.attendance import AttendanceLogger
from utils.privacy import redact_unknown_faces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()
_pipeline: UnifiedFaceSecurityPipeline | None = None
_attendance: AttendanceLogger | None = None


class FrameAnalysis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    faces: list[dict[str, Any]]


def get_pipeline() -> UnifiedFaceSecurityPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline


async def lifespan(app: FastAPI):
    global _pipeline, _attendance
    if settings.skip_model_init:
        logger.warning("SKIP_MODEL_INIT=1 — API starts without models (tests/dev).")
        yield
        return

    # Make startup forgiving for first-time users:
    # - If default weights path is missing, fall back to a COCO-pretrained YOLO checkpoint name
    #   so the server can still boot (even though it won't be a proper face detector until fine-tuned).
    # - If a custom weights path is configured and missing, fail with actionable guidance.
    yolo_weights = settings.yolo_weights
    if not yolo_weights.exists():
        if yolo_weights.as_posix().endswith("weights/yolov8_face.pt"):
            logger.warning(
                "YOLO weights not found at '%s'. Falling back to 'yolov8n.pt' so the API can start. "
                "For real face detection you must train/fine-tune and place your checkpoint at weights/yolov8_face.pt, "
                "or set YOLO_WEIGHTS to a valid path.",
                yolo_weights,
            )
            yolo_weights = Path("yolov8n.pt")
        else:
            raise RuntimeError(
                f"YOLO weights not found at '{yolo_weights}'. "
                "Fix YOLO_WEIGHTS (or place weights/yolov8_face.pt), or set SKIP_MODEL_INIT=true to start without models."
            )

    cfg = PipelineConfig(
        yolo_weights=yolo_weights,
        antispoof_weights=settings.antispoof_weights,
        chroma_dir=settings.chroma_dir,
        device=settings.torch_device,
        yolo_conf=settings.yolo_conf,
        yolo_iou=settings.yolo_iou,
        cosine_similarity_threshold=settings.cosine_similarity_threshold,
        antispoof_live_prob_min=settings.antispoof_live_prob_min,
        max_faces_per_frame=settings.max_faces_per_frame,
        mask_yolo_weights=settings.mask_yolo_weights,
    )
    _pipeline = UnifiedFaceSecurityPipeline(cfg, video_mesh=True)

    if settings.attendance_database_url:
        _attendance = AttendanceLogger(settings.attendance_database_url)
        logger.info("Attendance logging enabled (PostgreSQL).")

    yield

    if _pipeline is not None:
        _pipeline.close()


app = FastAPI(title="Face Security Pipeline", lifespan=lifespan)

# Built-in minimal UI (no frontend build step).
STATIC_DIR = ROOT / "api" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> Any:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return StreamingResponse(index_file.open("rb"), media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/infer/image", response_model=FrameAnalysis)
async def infer_image(file: UploadFile = File(...)) -> FrameAnalysis:
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    pipe = get_pipeline()
    faces = pipe.analyze_frame(frame)

    if _attendance:
        for f in faces:
            if f.get("known"):
                _attendance.log(
                    identity=str(f["identity"]),
                    similarity=float(f.get("verification_similarity") or 0.0),
                    liveness_prob=float(f.get("liveness_prob") or 0.0),
                    source="rest:/v1/infer/image",
                )

    return FrameAnalysis(faces=faces)


@app.post("/v1/enroll")
async def enroll(identity: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    """Register one embedding for `identity` using a frontal-ish crop image."""
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    pipe = get_pipeline()
    faces = pipe.analyze_frame(frame)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected for enrollment")

    best = max(faces, key=lambda x: float(x.get("detection_confidence") or 0.0))
    crop = pipe._crop_face(frame, best["bbox"])  # noqa: SLF001 — intentional helper reuse
    aligned = align_safe(pipe, crop)

    emb = pipe.embedder.embed_bgr_np([aligned], normalize=True)[0]
    pipe.index.upsert(identity=identity, embedding=emb)

    return {"identity": identity, "stored": True}


def align_safe(pipe: UnifiedFaceSecurityPipeline, crop_bgr: np.ndarray) -> np.ndarray:
    from utils.alignment import align_face_bgr_to_160, fallback_square_crop

    aligned = align_face_bgr_to_160(crop_bgr, pipe._mesh_static)  # noqa: SLF001
    return aligned if aligned is not None else fallback_square_crop(crop_bgr)


@app.get("/v1/stream/mjpeg")
async def mjpeg_stream():
    pipe = get_pipeline()

    async def boundaries():
        loop = asyncio.get_event_loop()

        def read_frame():
            cap = cv2.VideoCapture(settings.camera_source)
            ok, fr = cap.read()
            cap.release()
            return ok, fr

        while True:
            ok, frame = await loop.run_in_executor(None, read_frame)
            if not ok or frame is None:
                await asyncio.sleep(0.05)
                continue

            faces = pipe.analyze_frame(frame)

            if settings.redact_unknown:
                frame = redact_unknown_faces(frame, faces, blur_sigma=settings.blur_sigma)

            for f in faces:
                x1, y1, x2, y2 = [int(round(v)) for v in f["bbox"]]
                color = (0, 200, 0) if f.get("known") else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{f.get('identity')} | live={f.get('liveness_prob')}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            ok_j, buf = cv2.imencode(".jpg", frame)
            if not ok_j:
                await asyncio.sleep(0.01)
                continue
            chunk = (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            yield chunk
            await asyncio.sleep(0.03)

    return StreamingResponse(boundaries(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/stream/meta")
async def ws_meta(ws: WebSocket):
    """Lightweight JSON metadata channel (no raw biometric blobs leave process)."""
    await ws.accept()
    pipe = get_pipeline()
    cap = cv2.VideoCapture(settings.camera_source)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.02)
                continue
            faces = pipe.analyze_frame(frame)
            payload = {
                "faces": [
                    {
                        "bbox": f["bbox"],
                        "identity": f["identity"],
                        "known": f["known"],
                        "detection_confidence": f["detection_confidence"],
                        "verification_similarity": f.get("verification_similarity"),
                        "liveness_prob": f.get("liveness_prob"),
                        "mask_label": f.get("mask_label"),
                    }
                    for f in faces
                ]
            }
            await ws.send_json(payload)
            await asyncio.sleep(0.05)
    finally:
        cap.release()
        await ws.close()


@app.post("/v1/frame/save-redacted")
async def save_redacted(file: UploadFile = File(...)) -> dict[str, Any]:
    """Save under data/captures/ with unknown-face redaction (never arbitrary filesystem paths)."""
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    pipe = get_pipeline()
    faces = pipe.analyze_frame(frame)
    out = redact_unknown_faces(frame, faces, blur_sigma=settings.blur_sigma) if settings.redact_unknown else frame
    out_dir = ROOT / "data" / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{uuid4().hex}.jpg"
    ok = cv2.imwrite(str(fname), out)
    return {"written": bool(ok), "path": str(fname.relative_to(ROOT))}
