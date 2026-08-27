"""Download the public pose detector during a Render build."""

import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / ".ultralytics"))
from ultralytics import YOLO


target = Path("yolov8n-pose.pt")
if target.is_file():
    print(f"Pose detector already present: {target}")
else:
    print("Downloading yolov8n-pose.pt for image inference...")
    model = YOLO("yolov8n-pose.pt")
    source = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else target
    if source.resolve() != target.resolve() and source.is_file():
        target.write_bytes(source.read_bytes())
    if not target.is_file():
        raise RuntimeError(f"Ultralytics did not produce {target}")
    print(f"Pose detector ready: {target}")
