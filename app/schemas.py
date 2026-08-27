"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MovementMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speed: float
    distance: float


class BoxingMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_arm_extension: float
    right_arm_extension: float
    power_index: float
    guard_height: float


class PoseMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body_alignment: float
    stance_width: float


class MotionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    movement: MovementMetrics
    boxing_specific: BoxingMetrics
    pose: PoseMetrics


class FrameInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[float] | None = None
    detection: list[float] | None = None
    keypoints: list[float] | None = None
    bbox: list[float] | None = None
    motion_metrics: MotionMetrics | None = None

    @model_validator(mode="after")
    def one_frame_representation(self) -> "FrameInput":
        has_features = self.features is not None
        has_detection = self.detection is not None
        has_parts = self.keypoints is not None or self.bbox is not None
        if sum((has_features, has_detection, has_parts)) != 1:
            raise ValueError("provide exactly one of features, detection, or keypoints+bbox")
        if has_parts and (self.keypoints is None or self.bbox is None):
            raise ValueError("keypoints and bbox must be supplied together")
        if has_features and self.motion_metrics is not None:
            raise ValueError("motion_metrics is only valid with detection or keypoints+bbox")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: list[FrameInput] | None = Field(default=None, max_length=16)
    features: list[list[float]] | None = Field(default=None, max_length=16)
    top_k: int = Field(default=3, ge=1, le=8)
    include_attention: bool = False

    @model_validator(mode="after")
    def one_sequence_representation(self) -> "PredictRequest":
        if (self.frames is None) == (self.features is None):
            raise ValueError("provide exactly one of frames or features")
        return self


class Prediction(BaseModel):
    index: int
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    model: str
    checkpoint_sha256: str
    prediction: Prediction
    top_k: list[Prediction]
    sequence_length: int
    feature_dim: int
    inference_ms: float
    warm_start: bool
    attention: list[float] | None = None
