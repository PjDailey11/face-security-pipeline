# Development log

Maintainers: append a new dated section after every major change (architecture, training defaults, API contracts, privacy posture).

## 2026-05-07 — GitHub documentation (`README.md`)

**What changed**

- Added root `README.md` with app description, architecture overview (Mermaid), repository layout, local and Docker setup, training workflows, environment variable table, API reference, privacy notes, and links to `DEVLOG.md`.
- Made API startup more forgiving when default YOLO weights are missing (falls back to `yolov8n.pt` with a warning).

**Why**

- Give contributors and operators a single entry point for cloning, configuring weights, running the API, and understanding privacy boundaries.
- Prevent first-run crashes when users haven’t trained/copied `weights/yolov8_face.pt` yet.

**Next steps**

- Add an explicit `LICENSE` if the repo should be reusable by third parties.
- Optionally wire `cosine_similarity_threshold` / `antispoof_live_prob_min` to env-only overrides if operators must tune without code edits.

## 2026-05-06 — Initial end-to-end scaffold

**What changed**

- Added repository layout: `configs/`, `models/`, `training/`, `api/`, `utils/`, `tests/`, plus Docker assets.
- Documented detector hyperparameters in `configs/hyperparameters.yaml` (YOLOv8 transfer defaults: LR schedule via `lr0`/`lrf`, mosaic close-out, freeze layers, augmentation knobs).
- Implemented Ultralytics training driver `training/train_yolov8_face.py` with `freeze=N` backbone freezing and COCO seed weights (`yolov8n.pt` by default).
- Added dataset ingestion splitter `training/dataset_ingest.py` and Albumentations-heavy preprocessing helpers `training/preprocess.py` for lighting/occlusion/geometry diversity.
- Implemented anti-spoof CNN `models/antispoof_cnn.py` with micro-texture stem + training script `training/train_antispoof.py` (live vs spoof folders).
- Wired detection (`models/yolo_face.py`), landmark alignment (`utils/alignment.py` via MediaPipe FaceMesh), embeddings (`models/embedding_wrapper.py`, InceptionResnetV1 / FaceNet-style 512-D), and local Chroma cosine index (`utils/vector_store.py`).
- Built unified FastAPI service (`api/main.py`, `api/inference_pipeline.py`): REST image inference, multipart enrollment, MJPEG preview stream, WebSocket metadata-only stream, redacted disk capture under `data/captures/`.
- Optional PostgreSQL attendance logging (`utils/attendance.py`, env `ATTENDANCE_DATABASE_URL`) and optional mask overlay model hook (`utils/mask_detection.py`).
- Containerized with `Dockerfile` + `docker-compose.yml` (weights and DB volumes expected at runtime).

**Why**

- Deliver a privacy-by-design baseline: embeddings stay on-device in Chroma; API payloads avoid sending raw embedding vectors off-process; unknown faces can be blurred before persistence.
- Separate concerns so teams can train detectors/classifiers independently while keeping one inference orchestration layer.

**Next steps**

1. Prepare a labeled face-detection set (YOLO `datasets/faces/` layout) and run `training/train_yolov8_face.py`; export best weights to `weights/yolov8_face.pt`.
2. Curate `data/antispoof/live` and `data/antispoof/spoof` (prints, replays, masks) and train `training/train_antispoof.py`.
3. Tune `cosine_similarity_threshold` and `antispoof_live_prob_min` on a validation split; document chosen operating points here.
4. If GPU inference is required in Docker, add NVIDIA runtime flags and switch `TORCH_DEVICE` accordingly.
5. After each subsequent merge-worthy change, append a new dated section (what / why / next).

**Operational notes**

- First run of `facenet-pytorch` InceptionResnetV1 may download pretrained checkpoints from the internet (one-time artifact fetch, not runtime biometric exfiltration).
- For smoke-testing the API without GPUs or weights, set `SKIP_MODEL_INIT=true` (see `tests/test_health.py`).
