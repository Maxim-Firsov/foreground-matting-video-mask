from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


Roi = Tuple[int, int, int, int]
FrameSize = Tuple[int, int]

MORPH_KERNEL_SIZE = 5
OVERLAY_ALPHA = 0.35
ECC_ITERATIONS = 50
ECC_EPSILON = 1e-4
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 24.0
SUPPORT_DECAY = 0.82
MAX_COMPONENT_ASPECT = 5.5
MIN_FILL_RATIO = 0.10
MIN_SOLIDITY = 0.30


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for foreground extraction."""

    threshold: float
    downscale: float
    fps_override: float | None
    stabilize: bool
    keep_blobs: int
    min_area: int
    ema: float
    roi: Roi | None


@dataclass(frozen=True)
class VideoOutputs:
    """Output paths produced by the processor."""

    mask_path: Path
    overlay_path: Path


@dataclass
class MotionState:
    """State carried across frames."""

    prev_gray: np.ndarray
    mask_ema: np.ndarray | None = None
    temporal_support: np.ndarray | None = None


@dataclass(frozen=True)
class ComponentCandidate:
    """Connected foreground region scored by generic shape quality."""

    mask: np.ndarray
    area: int
    score: float


def parse_roi(roi_text: str | None) -> Roi | None:
    """Parse an ROI string in x,y,w,h form."""
    if roi_text is None:
        return None

    parts = [part.strip() for part in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be in x,y,w,h format.")

    try:
        x, y, w, h = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--roi must contain integer values.") from exc

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("--roi requires x,y >= 0 and w,h > 0.")

    return x, y, w, h


def validate_roi(roi: Roi | None, frame_size: FrameSize) -> Roi | None:
    """Ensure the ROI fits within the frame."""
    if roi is None:
        return None

    frame_width, frame_height = frame_size
    x, y, w, h = roi
    if x + w > frame_width or y + h > frame_height:
        raise ValueError(
            f"ROI {roi} exceeds frame bounds {frame_width}x{frame_height}."
        )
    return roi


def resize_frame(frame: np.ndarray, downscale: float) -> np.ndarray:
    """Resize the frame when downscale < 1.0; otherwise return it unchanged."""
    if downscale >= 1.0:
        return frame

    height, width = frame.shape[:2]
    target_width = max(1, int(width * downscale))
    target_height = max(1, int(height * downscale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def create_writer(output_path: Path, fps: float, frame_size: FrameSize) -> cv2.VideoWriter:
    """Create an MP4 writer and fail fast when initialization fails."""
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    return writer


def prepare_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to blurred grayscale for stabilization."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def estimate_ecc_warp(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray | None:
    """Estimate an affine warp aligning the current frame to the previous frame."""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_ITERATIONS,
        ECC_EPSILON,
    )

    try:
        cv2.findTransformECC(
            templateImage=prev_gray,
            inputImage=curr_gray,
            warpMatrix=warp_matrix,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
        )
    except cv2.error:
        return None

    return warp_matrix


def warp_frame_to_previous(frame: np.ndarray, warp_matrix: np.ndarray | None) -> np.ndarray:
    """Warp the current frame into the previous frame coordinate system."""
    if warp_matrix is None:
        return frame

    height, width = frame.shape[:2]
    return cv2.warpAffine(
        frame,
        warp_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def create_foreground_model() -> cv2.BackgroundSubtractor:
    """Create a foreground model that handles dynamic backgrounds better than raw flow."""
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )


def clean_foreground_mask(mask: np.ndarray) -> np.ndarray:
    """Clean up raw foreground output from the background subtractor."""
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    cleaned = cv2.medianBlur(mask, 5)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def smooth_mask(mask: np.ndarray, previous_ema: np.ndarray | None, ema: float) -> np.ndarray:
    """Smooth the foreground mask over time with EMA if requested."""
    current = (mask > 0).astype(np.float32)
    if ema <= 0.0:
        return current
    if previous_ema is None or previous_ema.shape != current.shape:
        return current
    return (ema * current) + ((1.0 - ema) * previous_ema)


def build_temporal_mask(
    smoothed_mask: np.ndarray,
    previous_support: np.ndarray | None,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Require detections to persist for multiple frames before surviving."""
    if previous_support is None or previous_support.shape != smoothed_mask.shape:
        support = smoothed_mask
    else:
        support = (previous_support * SUPPORT_DECAY) + smoothed_mask

    binary = np.where(support >= threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary, support


def score_component(mask: np.ndarray, area: int, bbox: Roi) -> ComponentCandidate | None:
    """Score a component by generic object-likeness rather than clip-specific cues."""
    _, _, w, h = bbox
    if min(w, h) < 6:
        return None

    bbox_area = max(w * h, 1)
    fill_ratio = float(area / bbox_area)
    aspect_ratio = float(max(w / max(h, 1), h / max(w, 1)))
    if fill_ratio < MIN_FILL_RATIO or aspect_ratio > MAX_COMPONENT_ASPECT:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hull_area = float(cv2.contourArea(cv2.convexHull(contours[0])))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    if solidity < MIN_SOLIDITY:
        return None

    score = (area * 0.01) + (fill_ratio * 2.0) + (solidity * 1.5) - (
        max(aspect_ratio - 1.0, 0.0) * 0.4
    )
    return ComponentCandidate(mask=mask, area=area, score=score)


def select_components(mask: np.ndarray, keep_blobs: int, min_area: int) -> np.ndarray:
    """Keep the strongest coherent components from the foreground mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[ComponentCandidate] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        component_mask = np.zeros_like(mask)
        component_mask[labels == label] = 255
        candidate = score_component(component_mask, area, (x, y, w, h))
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return np.zeros_like(mask)

    candidates.sort(key=lambda candidate: (candidate.score, candidate.area), reverse=True)
    output_mask = np.zeros_like(mask)
    for candidate in candidates[:keep_blobs]:
        output_mask = cv2.bitwise_or(output_mask, candidate.mask)
    return output_mask


def paste_roi_mask(mask_roi: np.ndarray, frame_size: FrameSize, roi: Roi | None) -> np.ndarray:
    """Paste an ROI-sized mask back into full-frame coordinates."""
    if roi is None:
        return mask_roi

    frame_width, frame_height = frame_size
    x, y, w, h = roi
    full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    full_mask[y : y + h, x : x + w] = mask_roi[:h, :w]
    return full_mask


def create_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend the foreground mask onto the original frame."""
    overlay_color = np.zeros_like(frame)
    overlay_color[..., 1] = mask
    return cv2.addWeighted(frame, 1.0, overlay_color, OVERLAY_ALPHA, 0.0)


class MotionMaskProcessor:
    """Foreground mask generator built around stabilization and background subtraction."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.foreground_model = create_foreground_model()

    def _output_fps(self, capture: cv2.VideoCapture) -> float:
        input_fps = capture.get(cv2.CAP_PROP_FPS)
        output_fps = self.config.fps_override if self.config.fps_override is not None else input_fps
        return output_fps if output_fps and output_fps > 0 else 30.0

    def process(self, input_path: Path, out_dir: Path) -> VideoOutputs:
        """Process a video and write mask and overlay MP4s."""
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / "mask.mp4"
        overlay_path = out_dir / "overlay.mp4"

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            capture.release()
            raise RuntimeError(f"No frames found in video: {input_path}")

        first_frame = resize_frame(first_frame, self.config.downscale)
        frame_height, frame_width = first_frame.shape[:2]
        frame_size = (frame_width, frame_height)
        roi = validate_roi(self.config.roi, frame_size)

        output_fps = self._output_fps(capture)
        mask_writer = create_writer(mask_path, output_fps, frame_size)
        overlay_writer = create_writer(overlay_path, output_fps, frame_size)

        state = MotionState(prev_gray=prepare_gray(first_frame))
        warmup_frame = first_frame if roi is None else first_frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        self.foreground_model.apply(warmup_frame, learningRate=1.0)

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                frame = resize_frame(frame, self.config.downscale)
                if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

                curr_gray = prepare_gray(frame)
                aligned_frame = frame
                if self.config.stabilize:
                    warp_matrix = estimate_ecc_warp(state.prev_gray, curr_gray)
                    aligned_frame = warp_frame_to_previous(frame, warp_matrix)
                    curr_gray = prepare_gray(aligned_frame)

                if roi is None:
                    model_frame = aligned_frame
                else:
                    x, y, w, h = roi
                    model_frame = aligned_frame[y : y + h, x : x + w]

                raw_mask = self.foreground_model.apply(model_frame, learningRate=-1)
                cleaned_mask = clean_foreground_mask(raw_mask)
                smoothed_mask = smooth_mask(cleaned_mask, state.mask_ema, self.config.ema)
                state.mask_ema = smoothed_mask

                temporal_mask, state.temporal_support = build_temporal_mask(
                    smoothed_mask,
                    state.temporal_support,
                    self.config.threshold,
                )
                selected_mask = select_components(
                    temporal_mask,
                    self.config.keep_blobs,
                    self.config.min_area,
                )
                full_mask = paste_roi_mask(selected_mask, frame_size, roi)

                mask_writer.write(cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR))
                overlay_writer.write(create_overlay(frame, full_mask))
                state.prev_gray = curr_gray
        finally:
            capture.release()
            mask_writer.release()
            overlay_writer.release()

        return VideoOutputs(mask_path=mask_path, overlay_path=overlay_path)
