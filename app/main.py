"""FastAPI application for Render or any ASGI-compatible Python host."""

from __future__ import annotations

import logging
import os
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import File, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.model import CLASS_NAMES, FEATURE_DIM, SEQUENCE_LENGTH, ModelBundle, load_model_bundle
from app.detector import PoseDetector
from app.preprocessing import InputContractError, build_feature_matrix, build_from_features
from app.sessions import SessionManager
from app.startup import StartupStatus
from app.schemas import PredictRequest, Prediction, PredictionResponse


logger = logging.getLogger("boxing-service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
ROOT = Path(__file__).resolve().parent.parent
STATIC_INDEX = ROOT / "static" / "index.html"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "best_boxing_model.pth")))
MODEL_NAME = os.getenv("MODEL_NAME", "boxing-action-lstm")
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "yolov8n-pose.pt")
POSE_CONFIDENCE = float(os.getenv("POSE_CONFIDENCE", "0.35"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "5000000"))


async def initialize_models(app: FastAPI) -> None:
    """Load both models in a worker thread while exposing progress to health/UI."""
    startup: StartupStatus = app.state.startup
    started = time.perf_counter()
    try:
        startup.update("loading_classifier", "Loading boxing classifier checkpoint")
        logger.info("STARTUP phase=loading_classifier path=%s", MODEL_PATH)
        bundle = await asyncio.to_thread(load_model_bundle, MODEL_PATH)
        logger.info(
            "STARTUP phase=classifier_ready device=%s warmup_ms=%.2f sha256=%s elapsed_ms=%.2f",
            bundle.device,
            bundle.warmup_ms,
            bundle.checkpoint_sha256,
            (time.perf_counter() - started) * 1000.0,
        )

        startup.update("loading_pose_detector", f"Loading pose detector {POSE_MODEL_PATH}")
        logger.info("STARTUP phase=loading_pose_detector path=%s", POSE_MODEL_PATH)
        detector_started = time.perf_counter()
        detector = await asyncio.to_thread(PoseDetector, POSE_MODEL_PATH, POSE_CONFIDENCE)
        logger.info(
            "STARTUP phase=pose_detector_ready detector_init_ms=%.2f elapsed_ms=%.2f",
            (time.perf_counter() - detector_started) * 1000.0,
            (time.perf_counter() - started) * 1000.0,
        )

        app.state.bundle = bundle
        app.state.detector = detector
        app.state.sessions = SessionManager(
            ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "900")),
            max_sessions=int(os.getenv("MAX_SESSIONS", "1000")),
        )
        startup.update("ready", "Classifier and pose detector are warm and ready")
        logger.info(
            "STARTUP phase=ready total_startup_ms=%.2f model_warmup_ms=%.2f",
            (time.perf_counter() - started) * 1000.0,
            bundle.warmup_ms,
        )
    except Exception as exc:
        startup.fail(f"Startup failed: {exc}")
        logger.exception("STARTUP phase=failed elapsed_ms=%.2f", (time.perf_counter() - started) * 1000.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.startup = StartupStatus()
    app.state.bundle = None
    app.state.detector = None
    app.state.sessions = None
    app.state.initializer = asyncio.create_task(initialize_models(app))
    yield
    app.state.initializer.cancel()
    await asyncio.gather(app.state.initializer, return_exceptions=True)
    app.state.bundle = None
    app.state.detector = None
    app.state.sessions = None


app = FastAPI(
    title="Boxing Action Inference API",
    version="1.0.0",
    description="Warm-started inference for the trained 16-frame boxing-action LSTM.",
    lifespan=lifespan,
)

cors_origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_INDEX)


@app.get("/health/live", tags=["health"])
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
def ready(request: Request) -> dict[str, str]:
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {"status": "ready"}


@app.get("/health/status", tags=["health"])
def status(request: Request) -> dict[str, object]:
    startup: StartupStatus = request.app.state.startup
    result = startup.snapshot()
    result["classifier_loaded"] = getattr(request.app.state, "bundle", None) is not None
    result["pose_detector_loaded"] = getattr(request.app.state, "detector", None) is not None
    return result


@app.get("/v1/metadata", tags=["model"])
def metadata(request: Request) -> dict[str, object]:
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {
        "model": MODEL_NAME,
        "classes": list(CLASS_NAMES),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
        "coordinate_contract": "raw detector pixel coordinates",
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "device": str(bundle.device),
        "warm_start": True,
        "image_inference": True,
        "pose_detector": POSE_MODEL_PATH,
    }


@app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PredictRequest, request: Request) -> PredictionResponse:
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    try:
        feature_matrix = (
            build_feature_matrix([frame.model_dump(exclude_none=True) for frame in payload.frames])
            if payload.frames is not None
            else build_from_features(payload.features)
        )
    except InputContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _classify(bundle, feature_matrix, payload.top_k, payload.include_attention)


def _classify(
    bundle: ModelBundle, feature_matrix: np.ndarray, top_k: int, include_attention: bool
) -> PredictionResponse:
    logits, attention, inference_ms = bundle.predict(feature_matrix[np.newaxis, ...])
    probabilities = torch_softmax(logits[0]).tolist()
    indices = np.argsort(np.asarray(probabilities))[::-1][:top_k]
    predictions = [
        Prediction(index=int(index), label=CLASS_NAMES[int(index)], confidence=float(probabilities[int(index)]))
        for index in indices
    ]
    attention_values = attention[0, :, 0].detach().cpu().tolist() if include_attention else None
    return PredictionResponse(
        model=MODEL_NAME,
        checkpoint_sha256=bundle.checkpoint_sha256,
        prediction=predictions[0],
        top_k=predictions,
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=FEATURE_DIM,
        inference_ms=round(inference_ms, 3),
        warm_start=True,
        attention=attention_values,
    )


@app.post("/v1/predict/frame", tags=["webcam"])
async def predict_frame(
    request: Request,
    frame: UploadFile = File(...),
    session_id: str = Header(..., alias="X-Session-ID"),
    top_k: int = 3,
    include_attention: bool = False,
) -> dict[str, object]:
    """Detect one webcam image, append it to a client session, and classify at 16 frames."""
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    detector: PoseDetector | None = getattr(request.app.state, "detector", None)
    sessions: SessionManager | None = getattr(request.app.state, "sessions", None)
    if bundle is None or detector is None or sessions is None:
        raise HTTPException(status_code=503, detail="service is not ready")
    if not 1 <= top_k <= len(CLASS_NAMES):
        raise HTTPException(status_code=422, detail=f"top_k must be between 1 and {len(CLASS_NAMES)}")

    try:
        session = sessions.get(session_id)
        image_bytes = await frame.read()
        detection = detector.detect(image_bytes, max_bytes=MAX_IMAGE_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if detection.detection is None:
        # Do not blend frames from two separate people or from a long occlusion.
        session.reset()
        logger.debug("FRAME session=%s status=no_detection detect_ms=%.2f", session_id, detection.detect_ms)
        return {
            "status": "no_detection",
            "session_id": session_id,
            "buffered_frames": 0,
            "required_frames": SEQUENCE_LENGTH,
            "detect_ms": round(detection.detect_ms, 3),
            "warm_start": True,
        }

    sequence = session.push(detection.detection)
    if sequence is None:
        logger.debug(
            "FRAME session=%s status=buffering buffered=%d/%d detect_ms=%.2f",
            session_id,
            len(session.features),
            SEQUENCE_LENGTH,
            detection.detect_ms,
        )
        return {
            "status": "buffering",
            "session_id": session_id,
            "buffered_frames": len(session.features),
            "required_frames": SEQUENCE_LENGTH,
            "detect_ms": round(detection.detect_ms, 3),
            "warm_start": True,
        }

    result = _classify(bundle, sequence, top_k, include_attention).model_dump()
    result.update({
        "status": "prediction",
        "session_id": session_id,
        "buffered_frames": SEQUENCE_LENGTH,
        "detect_ms": round(detection.detect_ms, 3),
        "bbox": detection.bbox.tolist() if detection.bbox is not None else None,
        "keypoints": detection.keypoints.tolist() if detection.keypoints is not None else None,
    })
    logger.info(
        "INFERENCE session=%s label=%s confidence=%.4f detect_ms=%.2f inference_ms=%.2f",
        session_id,
        result["prediction"]["label"],
        result["prediction"]["confidence"],
        detection.detect_ms,
        result["inference_ms"],
    )
    return result


@app.post("/v1/reset", tags=["webcam"])
def reset_session(request: Request, session_id: str = Header(..., alias="X-Session-ID")) -> dict[str, object]:
    sessions: SessionManager | None = getattr(request.app.state, "sessions", None)
    if sessions is None:
        raise HTTPException(status_code=503, detail="service is not ready")
    try:
        sessions.get(session_id).reset()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "reset", "session_id": session_id}


def torch_softmax(logits):
    import torch

    return torch.softmax(logits, dim=0).detach().cpu()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
