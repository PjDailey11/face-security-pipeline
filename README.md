# Face Security Pipeline

An **on-device** face detection and identity-verification stack aimed at privacy-conscious deployments: **YOLOv8** face localization (fine-tuned from COCO-pretrained weights), **MediaPipe** landmark alignment, **512-D embeddings** (InceptionResnetV1 via [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch)), **local vector search** (ChromaDB + cosine similarity), an optional **anti-spoof CNN**, and a **FastAPI** service for images and camera streams.

**Biometric embeddings stay local.** There are no API endpoints that return raw embedding vectors to clients; optional PostgreSQL logging stores metadata only (identity labels and scores), not embeddings.

---

## Features

| Capability | Details |
|------------|---------|
| Face detection | Ultralytics **YOLOv8**, transfer learning with backbone freeze (`freeze=N`) |
| Alignment | Similarity warp from FaceMesh landmarks; fallback square crop |
| Verification | Cosine similarity against enrolled identities in **ChromaDB** |
| Liveness (optional) | Binary CNN emphasizing micro-texture (`weights/antispoof.pt`) |
| Streaming | **MJPEG** preview + **WebSocket** metadata (boxes, identity, scores—no embeddings) |
| Privacy | Blur **unknown** faces before save; captures written only under `data/captures/` |
| Optional modules | **PostgreSQL** attendance (`ATTENDANCE_DATABASE_URL`); secondary **YOLO** for mask overlays |

---

## Architecture

```mermaid
flowchart LR
  subgraph ingest [Input]
    IMG[Image / Camera]
  end
  subgraph pipeline [Pipeline]
    YOLO[YOLOv8 faces]
    LM[MediaPipe align]
    EMB[InceptionResnetV1]
    AS[Anti-spoof CNN]
    IDX[(ChromaDB)]
  end
  IMG --> YOLO --> LM --> EMB --> IDX
  LM --> AS
  IDX -->|cosine| OUT[Identity + scores]
  AS --> OUT
```

Hyperparameters for training are documented in [`configs/hyperparameters.yaml`](configs/hyperparameters.yaml). Runtime paths and thresholds live in [`configs/pipeline.yaml`](configs/pipeline.yaml) (reference) and **environment variables** (actual overrides).

---

## Repository layout

```
face-security-pipeline/
├── api/                 # FastAPI app & unified inference pipeline
├── configs/             # YAML: hyperparameters, dataset layout, pipeline notes
├── models/              # YOLO wrapper, embedding net, anti-spoof CNN
├── training/            # YOLO & anti-spoof training + dataset helpers
├── utils/               # Alignment, vector index, privacy, stream, attendance, mask hook
├── tests/               # Pytest (smoke with SKIP_MODEL_INIT)
├── weights/             # Place *.pt checkpoints here (gitignored except .gitkeep)
├── data/                # Chroma persistence, redacted captures (gitignored)
├── DEVLOG.md            # Maintainer log (what / why / next)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Prerequisites

- **Python 3.11+** recommended (Dockerfile uses 3.11)
- **CUDA** optional but recommended for training and real-time inference
- **Webcam or video file path** if you use streaming endpoints (`CAMERA_SOURCE`)

---

## Quick start (local)

### 1. Clone

```bash
git clone https://github.com/PjDailey11/face-security-pipeline.git
cd face-security-pipeline
```

### 2. Virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy the example file and edit paths/device:

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Unix
```

See [Environment variables](#environment-variables) below.

### 5. Weights

Place checkpoints under `weights/`:

| File | Purpose |
|------|---------|
| `yolov8_face.pt` | Fine-tuned **face detector** (required for real use; COCO weights alone do not detect “face” class) |
| `antispoof.pt` | Optional; if missing, liveness scores are disabled (warning in logs) |

After training (below), copy your best run’s `weights/best.pt` to `weights/yolov8_face.pt`.

If you see `FileNotFoundError: weights\\yolov8_face.pt` on startup, you have three options:

- **Best**: train/fine-tune and copy your checkpoint to `weights/yolov8_face.pt`.
- **Quick boot**: set `YOLO_WEIGHTS=yolov8n.pt` (server boots, but it is **not** a face detector until fine-tuned).
- **Smoke test**: set `SKIP_MODEL_INIT=true` to start the API without loading models (health/docs only).

### 6. Run the API

From the **repository root** (so imports resolve):

```bash
python run.py
```

Open:

- **UI**: `http://127.0.0.1:8000/` (paste/drag/drop/capture images, then Enroll/Infer)
- **API docs**: `http://127.0.0.1:8000/docs`

---

## Docker

Build and run (CPU by default in `docker-compose.yml`):

```bash
docker compose build
docker compose up
```

Mount `./weights` and `./data` so checkpoints and Chroma survive restarts.

**GPU (optional):** install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), add a `deploy.resources.reservations.devices` GPU section to the `api` service, set `TORCH_DEVICE=cuda:0`, and use a CUDA-enabled PyTorch base or install `torch` with CUDA inside the image.

---

## Training

### Face detector (YOLOv8)

1. Prepare data in YOLO layout (see [`configs/dataset_face.yaml`](configs/dataset_face.yaml)):

   ```
   datasets/faces/images/train/
   datasets/faces/images/val/
   datasets/faces/labels/train/   # *.txt per image, normalized xywh
   datasets/faces/labels/val/
   ```

2. Optionally split a flat folder of images:

   ```bash
   python training/dataset_ingest.py --flat-dir path/to/images --out datasets/faces_prep
   ```

   Then add labels (YOLO format) before training.

3. Train (defaults pull hyperparameters from `configs/hyperparameters.yaml`):

   ```bash
   python training/train_yolov8_face.py --data configs/dataset_face.yaml --weights yolov8n.pt
   ```

4. Copy the best weights to `weights/yolov8_face.pt`.

### Anti-spoof CNN

1. Organize binary folders:

   ```
   data/antispoof/live/**   # genuine captures
   data/antispoof/spoof/**  # print / screen / replay attacks
   ```

2. Train:

   ```bash
   python training/train_antispoof.py --data data/antispoof --out weights/antispoof.pt
   ```

Augmentation ideas for detector/spoof robustness are in [`training/preprocess.py`](training/preprocess.py).

---

## Environment variables

Loaded from `.env` or the shell (see [`api/settings.py`](api/settings.py)). With **pydantic-settings**, environment keys are typically the **uppercase snake_case** form of each setting field name.

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_DEVICE` | `cuda:0` | Falls back to CPU in pipeline if CUDA unavailable |
| `YOLO_WEIGHTS` | `weights/yolov8_face.pt` | Detector checkpoint |
| `ANTISPOOF_WEIGHTS` | `weights/antispoof.pt` | Optional; if file missing, liveness is skipped |
| `CHROMA_DIR` | `data/chroma_faces` | Persistent Chroma directory |
| `MASK_YOLO_WEIGHTS` | _(unset)_ | Optional second YOLO for mask/class overlay |
| `ATTENDANCE_DATABASE_URL` | _(unset)_ | e.g. `postgresql+psycopg2://user:pass@host:5432/db` |
| `COSINE_SIMILARITY_THRESHOLD` | `0.65` | Minimum cosine similarity to accept an enrolled identity |
| `ANTISPOOF_LIVE_PROB_MIN` | `0.55` | Below this live probability, verification is rejected |
| `YOLO_CONF` / `YOLO_IOU` | `0.35` / `0.45` | Detector NMS thresholds |
| `MAX_FACES_PER_FRAME` | `20` | Cap faces processed per frame |
| `REDACT_UNKNOWN` | `true` | Blur unknown faces on MJPEG stream |
| `BLUR_SIGMA` | `25` | Blur strength |
| `CAMERA_SOURCE` | `0` | OpenCV camera index or video path string |
| `SKIP_MODEL_INIT` | `false` | Set `true` for tests (`pytest`) without loading models |

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/v1/infer/image` | Multipart image → JSON list of faces (bbox, identity, scores, liveness, mask hint) |
| `POST` | `/v1/enroll` | Form field `identity` + image file → stores one embedding in Chroma |
| `GET` | `/v1/stream/mjpeg` | Multipart MJPEG stream with overlays (local/LAN only recommended) |
| `WS` | `/ws/stream/meta` | JSON per frame: boxes + identity strings + scores (**no** embedding vectors) |
| `POST` | `/v1/frame/save-redacted` | Saves redacted JPEG under `data/captures/<uuid>.jpg` |

Example enrollment (`curl`):

```bash
curl -X POST "http://127.0.0.1:8000/v1/enroll" ^
  -F "identity=alice" ^
  -F "file=@photo.jpg"
```

---

## Tests

```bash
pytest
```

The suite uses `SKIP_MODEL_INIT=true` so the app starts without GPU/heavy checkpoints (see [`tests/test_health.py`](tests/test_health.py)).

---

## Privacy & security notes

- **Default posture:** inference and embedding storage are **local**; design goal is **no outbound transmission of biometric vectors**.
- **Streams still carry pixels** (MJPEG). Treat networks as sensitive; use TLS and authentication if exposed beyond localhost.
- **First run** of `facenet-pytorch` may download **public pretrained weights** (one-time).
- **Compliance:** facial recognition is regulated in many jurisdictions; obtain consent and legal review before production use.
- Enrolled identities are **opaque strings** plus vectors on disk (Chroma); protect filesystem and backups accordingly.

---

## Changelog / deeper notes

See [`DEVLOG.md`](DEVLOG.md) for architecture decisions and next steps.

---

## License

No license file is bundled yet. Add a `LICENSE` (e.g. MIT, Apache-2.0) before redistributing.
