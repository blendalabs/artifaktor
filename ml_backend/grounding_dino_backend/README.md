# Grounding DINO Label Studio ML backend

This backend provides automatic preannotations for:

- Rectangle label: `body_distortion`
- Per-region attribute (auto-filled when possible): `body_part`
  - `face`, `hair`, `hand`, `arm`, `leg`, `torso`, `full_body`, `other`
- Optional per-region attribute (manual): `severity`
  - `mild`, `medium`, `severe`

## Runtime modes

1. **Grounding DINO mode** (preferred)
   - Requires `torch` + `transformers`
   - Uses zero-shot detection model from Hugging Face

2. **OpenCV fallback mode**
   - Used automatically if Grounding DINO deps are unavailable
   - Returns conservative generic `body_distortion` boxes

## Start locally

From project root:

```bash
source .envrc
./scripts/start_ml_backend.sh
```

## Start Label Studio with warmup (recommended)

This starts (or reuses) the ML backend, runs a warmup predict call on a local frame,
and then starts Label Studio.

After `source .envrc`, use Label Studio normally:

```bash
label-studio start
```

(`.envrc` wraps `label-studio start` to call `scripts/start_label_studio.sh`.)

You can still call the launcher directly:

```bash
./scripts/start_label_studio.sh
```

The warmup removes the first-request model load delay in Label Studio.

## RTX 4080 Laptop (local GPU) setup

Install CUDA-enabled PyTorch wheels in `.venv`:

```bash
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
.venv/bin/python -m pip install transformers accelerate
```

Then run backend with:

```bash
source .envrc
./scripts/start_ml_backend.sh
```

`start_ml_backend.sh` sets `GROUNDING_DINO_DEVICE=cuda` by default and includes the NVIDIA driver path (`/run/opengl-driver/lib`) in `LD_LIBRARY_PATH`.

Health check:

```bash
curl http://127.0.0.1:9090/health
```

## Attach backend to Label Studio project

```bash
source .envrc
.venv/bin/python scripts/setup_label_studio_ml_backend.py \
  --url http://127.0.0.1:8080 \
  --username "$LABEL_STUDIO_USERNAME" \
  --password "$LABEL_STUDIO_PASSWORD" \
  --project-title "AI Video Artifact Detection" \
  --backend-url http://127.0.0.1:9090 \
  --test-predict
```

## Retrain loop in Label Studio (15-min cycle)

This backend now supports manual retraining from Label Studio's **Train** button:

1. Label for ~15 minutes
2. Click **Train** on the connected ML backend
3. In Data Manager, run **Retrieve Predictions**
4. Review/edit and continue

Training uses a lightweight frame-sequence KNN learner over accepted annotations
(frame index → center/size). This gives immediate iterative improvement for sequential frames.

## Tuning

Environment variables:

- `GROUNDING_DINO_MODEL_ID` (default: `IDEA-Research/grounding-dino-tiny`)
- `GROUNDING_DINO_DEVICE` (default in model: `cpu`; `scripts/start_ml_backend.sh` sets `cuda` if unset)
- `GROUNDING_DINO_BOX_THRESHOLD` (default: `0.22`)
- `GROUNDING_DINO_TEXT_THRESHOLD` (default: `0.20`)
- `GROUNDING_DINO_MAX_DETECTIONS` (default: `20`)

Sequence-learning settings:

- `SEQUENCE_LEARNING_ENABLED` (default: `true`)
- `SEQUENCE_PREFER_TRAINED` (default: `true`)
- `SEQUENCE_MIN_TRAIN_SAMPLES` (default: `8`)
- `SEQUENCE_K_NEIGHBORS` (default: `4`)
- `SEQUENCE_DEFAULT_BOX_WIDTH_PCT` (default: `6.0`)
- `SEQUENCE_DEFAULT_BOX_HEIGHT_PCT` (default: `10.0`)
- `SEQUENCE_TRAIN_ON_ANNOTATION_EVENTS` (default: `false`)

## Modal / cloud GPU note

Deploy this same backend in a GPU container and expose it over HTTPS.
Then set Label Studio backend URL to that public endpoint (`https://...`).
The app code in `_wsgi.py` + `model.py` is deployment-agnostic.
