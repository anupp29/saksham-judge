"""
test_local.py
-------------
Smoke-test the full pipeline on your webcam locally before deploying.

Usage:
    pip install torch ultralytics opencv-python gradio
    cp /path/to/best_boxing_model.pth .
    python test_local.py

Press 'q' to quit.
Press 'r' to reset the frame buffer.
"""

import cv2
import numpy as np
from inference import BoxingInferenceEngine
from app import draw_overlay


def main():
    print("Loading engine…")
    engine = BoxingInferenceEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try changing VideoCapture(0) to (1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Webcam open. Press 'q' to quit, 'r' to reset buffer.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # OpenCV gives BGR; convert to RGB for the engine
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = engine.predict(frame_rgb)

        overlay_rgb = draw_overlay(frame_rgb, result)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("Boxing Judge — local test", overlay_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            engine.reset()
            print("Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()