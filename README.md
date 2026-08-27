---
title: Boxing Judge
emoji: 🥊
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Boxing Judge — Real-Time Punch Classifier

Live boxing action recognition from your webcam.

## Pipeline

```
Webcam frame
    │
    ▼
YOLOv8-nano-pose  →  17 keypoints + bounding box
    │
    ▼
Feature extractor  →  63-dim vector per frame
    │
    ▼
Sliding window (16 frames)
    │
    ▼
Bidirectional LSTM + Attention  →  8-class softmax
    │
    ▼
Overlay + probability bars
```

## Classes

| Label | Description |
|-------|-------------|
| `jab_left` | Left jab |
| `jab_right` | Right jab |
| `hook_left` | Left hook |
| `hook_right` | Right hook |
| `uppercut_left` | Left uppercut |
| `uppercut_right` | Right uppercut |
| `body_hook_left` | Left body hook |
| `body_hook_right` | Right body hook |

## Usage

1. Allow browser webcam access.
2. Stand 2–3 m from the camera so your full upper body is visible.
3. Throw punches — predictions appear with a confidence bar and per-class probabilities.
4. Click **Reset frame buffer** if you want to clear the rolling window.

## Model details

- **Backbone**: YOLOv8-nano-pose (keypoint extraction)
- **Classifier**: 2-layer Bidirectional LSTM + temporal attention head
- **Input**: 63 features × 16 frames
  - 51 normalised keypoint coords (17 pts × x, y, conf)
  - 4 normalised bounding-box coords
  - 8 boxing-specific motion metrics
- **Parameters**: ~2.8 M
- **Inference**: ~15–25 fps on CPU (HF free tier)