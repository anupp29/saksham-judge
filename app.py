"""
Boxing Judge — Gradio app
/ping  → keep-alive JSON endpoint (hit with cron job to prevent sleep)
/      → full Gradio webcam UI
"""

from __future__ import annotations

import threading
import time

import cv2
import gradio as gr
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from feature_extractor import CLASS_NAMES
from inference import BoxingInferenceEngine

# ── skeleton ───────────────────────────────────────────────────────────────
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
COLOURS = {
    "jab_left":        (255,100,100), "jab_right":        (100,255,100),
    "hook_left":       (255,200,  0), "hook_right":       (  0,200,255),
    "uppercut_left":   (200,  0,255), "uppercut_right":   (255,150, 50),
    "body_hook_left":  ( 50,255,200), "body_hook_right":  (200, 50,255),
}


def draw_overlay(frame_rgb: np.ndarray, result: dict) -> np.ndarray:
    img = frame_rgb.copy()
    h, w = img.shape[:2]
    label, conf, fps = result["label"], result["confidence"], result["fps"]
    kp, probs = result.get("keypoints"), result.get("probabilities", {})

    # skeleton
    if kp is not None:
        pts = [(int(kp[i,0]), int(kp[i,1])) if kp[i,2] > 0.3 else None for i in range(17)]
        for a, b in SKELETON:
            if pts[a] and pts[b]:
                cv2.line(img, pts[a], pts[b], (100,220,100), 2)
        for p in pts:
            if p:
                cv2.circle(img, p, 4, (255,255,0), -1)

    # label bar
    colour = COLOURS.get(label, (200,200,200))
    cv2.rectangle(img, (0,0), (w,60), (0,0,0), -1)
    cv2.putText(img, label, (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, colour, 2, cv2.LINE_AA)
    cv2.rectangle(img, (10,50), (10+int(conf*(w//2)),58), colour, -1)
    cv2.putText(img, f"{fps:.1f} fps", (w-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)

    # prob strip
    bh = 12
    y0_base = h - len(CLASS_NAMES)*(bh+2) - 10
    for i, cls in enumerate(CLASS_NAMES):
        p  = probs.get(cls, 0.0)
        y0 = y0_base + i*(bh+2)
        cv2.rectangle(img, (0,y0), (int(p*200), y0+bh), COLOURS.get(cls,(200,200,200)), -1)
        cv2.putText(img, f"{cls} {p:.2f}", (205, y0+bh-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
    return img


# ── warm up engine at import time (happens inside Docker at container start) ──
print("[app] Pre-warming inference engine…")
_engine = BoxingInferenceEngine.get()
print("[app] Engine warm ✓")


# ── FastAPI app (Gradio mounts onto it) ───────────────────────────────────
fastapi_app = FastAPI()

@fastapi_app.get("/ping")
def ping():
    """
    Keep-alive endpoint.
    Hit this every 25 seconds from your cron job:
        curl https://YOUR-SPACE.hf.space/ping
    Returns 200 + uptime so you can monitor health too.
    """
    return JSONResponse({"status": "ok", "uptime_s": round(time.time() - _START, 1)})

_START = time.time()


# ── Gradio processing ─────────────────────────────────────────────────────
def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return np.zeros((480,640,3), dtype=np.uint8)
    result = _engine.predict(frame)
    return draw_overlay(frame, result)


def reset_buffer():
    _engine.reset()
    return "Buffer reset ✓"


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Boxing Judge", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🥊 Boxing Judge — Real-Time Punch Classifier")
    gr.Markdown("> Stand **2–3 m** from camera · upper body visible · allow webcam")

    with gr.Row():
        with gr.Column(scale=3):
            cam = gr.Image(sources=["webcam"], streaming=True,
                           type="numpy", mirror_webcam=False, label="Webcam")
            out = gr.Image(type="numpy", label="Overlay")
        with gr.Column(scale=1):
            gr.Markdown("### Classes\n" + "\n".join(f"• `{c}`" for c in CLASS_NAMES))
            gr.Markdown("---")
            btn   = gr.Button("🔄 Reset buffer", variant="secondary")
            status = gr.Textbox(label="Status", interactive=False)

    cam.stream(fn=process_frame, inputs=cam, outputs=out, time_limit=60)
    btn.click(fn=reset_buffer, outputs=status)

# Mount Gradio onto FastAPI so /ping and / coexist
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)