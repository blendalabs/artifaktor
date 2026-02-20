# Grounding DINO Label Studio ML backend

This backend provides automatic preannotations (bounding boxes) for the artifact labels:

- `flicker`
- `morph`
- `hand_distortion`
- `face_distortion`
- `temporal_blur`
- `edge_artifact`

## Runtime modes

1. **Grounding DINO mode** (preferred)
   - Requires `torch` + `transformers`
   - Uses zero-shot detection model from Hugging Face

2. **OpenCV fallback mode**
   - Used automatically if Grounding DINO deps are unavailable
   - Returns conservative `edge_artifact` boxes only

## Start locally

From project root:

```bash
source .envrc
./scripts/start_ml_backend.sh
```

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

## Tuning

Environment variables:

- `GROUNDING_DINO_MODEL_ID` (default: `IDEA-Research/grounding-dino-tiny`)
- `GROUNDING_DINO_DEVICE` (default: `cpu`, set `cuda` on GPU)
- `GROUNDING_DINO_BOX_THRESHOLD` (default: `0.22`)
- `GROUNDING_DINO_TEXT_THRESHOLD` (default: `0.20`)
- `GROUNDING_DINO_MAX_DETECTIONS` (default: `20`)

## Modal / cloud GPU note

Deploy this same backend in a GPU container and expose it over HTTPS.
Then set Label Studio backend URL to that public endpoint (`https://...`).
The app code in `_wsgi.py` + `model.py` is deployment-agnostic.
