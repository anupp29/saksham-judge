# Boxing action inference service

This repository serves `best_boxing_model.pth` as a warm-started FastAPI web service suitable for Render. The service loads the checkpoint and YOLO pose detector once during process startup, validates the classifier state-dict architecture, switches both models to inference mode, and warms both before readiness is reported.

## Model contract

The checkpoint was trained by the notebook in this repository with:

- sequence shape: `(16, 63)`
- 51 raw COCO-17 keypoint values, followed by 4 raw bbox values, followed by 8 motion values
- two explicit bidirectional LSTM blocks with hidden size 128 and attention
- eight classes, in the exact `labels.json` order

The notebook uses raw detector pixel coordinates. Do not normalize the first 55 values. A frame can be supplied as:

- `detection`: 55 values `[x1,y1,x2,y2, kp0_x,kp0_y,kp0_conf, ... kp16_x,kp16_y,kp16_conf]`; motion is computed server-side from adjacent frames, or
- `keypoints` (51) + `bbox` (4), with optional notebook-shaped `motion_metrics`, or
- `features`: a precomputed 63-value vector.

The API requires exactly 16 frames for the feature endpoint. This is deliberate: padding or silently selecting a window can change the model's output. The `/v1/predict/frame` endpoint accepts one JPEG/PNG/WebP webcam frame at a time, runs YOLOv8-nano-pose, keeps an isolated expiring buffer per `X-Session-ID`, and starts returning predictions after 16 accepted detections.

## Run locally

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 10000
```

Open `http://127.0.0.1:10000/` for the static test client. API documentation is at `/docs`.

The browser client uses the live webcam endpoint. For machine clients, the outer `features` list must contain 16 rows and every row must contain 63 finite numbers. A frame request is sent as multipart form data with a required `X-Session-ID` header.

## Deploy to Render

1. Push this repository, including `best_boxing_model.pth` and `yolov8n-pose.pt`, to the Git provider connected to Render.
2. Create a Blueprint using `render.yaml`, or create a Docker web service pointing at `Dockerfile`.
3. Set `CORS_ORIGINS` only if a separate frontend needs cross-origin access; leave it unset for the bundled same-origin client.
4. Verify `/health/ready`, `/v1/metadata`, and a known 16-frame request after deploy.

The service is intentionally configured with one worker. Multiple workers would load a separate model copy per process and defeat the single-process warm-start/memory budget. Scale horizontally with Render instances when needed.

The Render build downloads `yolov8n-pose.pt` into the service image. If the exact detector used to create the training `.npy` files is available, set `POSE_MODEL_PATH` to that artifact instead; using a different detector can change feature distributions and accuracy.
