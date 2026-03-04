from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


DEFAULT_THRESHOLD = 1.5
DEFAULT_DOWNSCALE = 1.0
DEFAULT_OUT_DIR = "outputs"
MORPH_KERNEL_SIZE = 5
OVERLAY_ALPHA = 0.35
DEFAULT_KEEP_BLOBS = 1
DEFAULT_MIN_AREA = 500
DEFAULT_EMA = 0.0
ECC_ITERATIONS = 50
ECC_EPSILON = 1e-4
FLOW_BLUR_SIZE = 7
COHERENCE_WINDOW_SIZE = 21
COMPONENT_TRACK_DILATION = 21
OVERLAP_SCORE_WEIGHT = 4.0
AREA_SCORE_WEIGHT = 1.0
TRACK_MAX_MISSES = 8
TRACK_MIN_HITS = 4
TRACK_IOU_WEIGHT = 6.0
TRACK_DISTANCE_WEIGHT = 2.5
TRACK_SHAPE_WEIGHT = 1.5
TRACK_MOTION_WEIGHT = 2.0
TRACK_PERSISTENCE_WEIGHT = 1.5
MAX_TRACK_DISTANCE_FACTOR = 0.2

Roi = Tuple[int, int, int, int]


@dataclass
class BlobCandidate:
    """Motion region candidate extracted from the thresholded mask."""

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int
    centroid: Tuple[float, float]
    compactness: float
    mean_motion: float


@dataclass
class MotionTrack:
    """Persistent object hypothesis across frames."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    compactness: float
    mean_motion: float
    hits: int = 1
    misses: int = 0
    age: int = 1

    @property
    def confidence(self) -> float:
        """Track confidence grows with persistence and decays with misses."""
        persistence = min(self.hits / max(TRACK_MIN_HITS, 1), 1.5)
        miss_penalty = max(0.0, 1.0 - (self.misses / max(TRACK_MAX_MISSES, 1)))
        return persistence * miss_penalty


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the optical flow mask pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate motion mask and overlay videos from dense optical flow."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory for output videos. Defaults to '{DEFAULT_OUT_DIR}'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Optical-flow magnitude threshold used to create the motion mask.",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=DEFAULT_DOWNSCALE,
        help="Optional frame downscale factor for faster processing. Must be > 0.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional output FPS override. If omitted, the input video FPS is preserved.",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable ECC-based camera motion compensation.",
    )
    parser.add_argument(
        "--keep-blobs",
        type=int,
        default=DEFAULT_KEEP_BLOBS,
        help="Keep the largest N connected regions in the mask after filtering.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Discard connected regions smaller than this many pixels.",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=DEFAULT_EMA,
        help="EMA factor for smoothing optical-flow magnitude before thresholding. Use 0 to disable.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI as x,y,w,h. Flow is computed only inside the ROI and pasted back into the full frame.",
    )
    return parser.parse_args()


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


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path, Roi | None]:
    """Validate CLI arguments and return normalized paths."""
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    roi = parse_roi(args.roi)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if args.threshold < 0:
        raise ValueError("--threshold must be >= 0.")
    if args.downscale <= 0:
        raise ValueError("--downscale must be > 0.")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0 when provided.")
    if args.keep_blobs < 1:
        raise ValueError("--keep-blobs must be >= 1.")
    if args.min_area < 0:
        raise ValueError("--min-area must be >= 0.")
    if not 0.0 <= args.ema <= 1.0:
        raise ValueError("--ema must be between 0.0 and 1.0.")

    return input_path, out_dir, roi


def create_writer(output_path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an MP4 writer and fail fast if initialization does not succeed."""
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


def resize_frame(frame: np.ndarray, downscale: float) -> np.ndarray:
    """Resize the frame when downscale < 1.0; otherwise return it unchanged."""
    if downscale >= 1.0:
        return frame

    height, width = frame.shape[:2]
    target_width = max(1, int(width * downscale))
    target_height = max(1, int(height * downscale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def validate_roi(roi: Roi | None, frame_size: Tuple[int, int]) -> Roi | None:
    """Ensure the ROI fits within the current frame size."""
    if roi is None:
        return None

    frame_width, frame_height = frame_size
    x, y, w, h = roi
    if x + w > frame_width or y + h > frame_height:
        raise ValueError(
            f"ROI {roi} exceeds frame bounds {frame_width}x{frame_height}."
        )
    return roi


def estimate_ecc_warp(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray | None:
    """Estimate an affine warp that maps the current frame onto the previous frame."""
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
    """Warp the current frame into the previous frame's coordinate system when a warp exists."""
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


def prepare_gray_for_flow(gray_frame: np.ndarray) -> np.ndarray:
    """Suppress fine texture so dense flow favors coherent object motion over water ripples."""
    return cv2.GaussianBlur(gray_frame, (FLOW_BLUR_SIZE, FLOW_BLUR_SIZE), 0)


def score_connected_components(
    labels: np.ndarray,
    stats: np.ndarray,
    keep_blobs: int,
    min_area: int,
    previous_mask: np.ndarray | None,
) -> set[int]:
    """Rank connected components by temporal consistency first, then by size."""
    candidates: list[Tuple[int, float]] = []
    overlap_reference: np.ndarray | None = None
    if previous_mask is not None and np.count_nonzero(previous_mask) > 0:
        tracking_kernel = np.ones(
            (COMPONENT_TRACK_DILATION, COMPONENT_TRACK_DILATION), dtype=np.uint8
        )
        overlap_reference = cv2.dilate(previous_mask, tracking_kernel)

    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = labels == label
        overlap_pixels = 0
        if overlap_reference is not None:
            overlap_pixels = int(np.count_nonzero(component_mask & (overlap_reference > 0)))

        score = (overlap_pixels * OVERLAP_SCORE_WEIGHT) + (area * AREA_SCORE_WEIGHT)
        candidates.append((label, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return {label for label, _ in candidates[:keep_blobs]}


def filter_mask_components(
    mask: np.ndarray,
    keep_blobs: int,
    min_area: int,
    previous_mask: np.ndarray | None,
) -> np.ndarray:
    """Keep temporally stable connected components above the minimum area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    kept_labels = score_connected_components(labels, stats, keep_blobs, min_area, previous_mask)
    if not kept_labels:
        return np.zeros_like(mask)

    filtered = np.zeros_like(mask)
    for label in kept_labels:
        filtered[labels == label] = 255

    return filtered


def bbox_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """Compute IoU between two axis-aligned boxes in x,y,w,h form."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def extract_blob_candidates(
    mask: np.ndarray,
    motion_response: np.ndarray,
    min_area: int,
) -> list[BlobCandidate]:
    """Extract geometric and motion features for connected motion regions."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[BlobCandidate] = []

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

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        perimeter = float(cv2.arcLength(contours[0], True))
        compactness = 0.0
        if perimeter > 0:
            compactness = float((4.0 * np.pi * area) / (perimeter * perimeter))

        component_pixels = component_mask > 0
        mean_motion = float(np.mean(motion_response[component_pixels])) if np.any(component_pixels) else 0.0

        centroid_x = float(centroids[label][0])
        centroid_y = float(centroids[label][1])
        candidates.append(
            BlobCandidate(
                mask=component_mask,
                bbox=(x, y, w, h),
                area=area,
                centroid=(centroid_x, centroid_y),
                compactness=compactness,
                mean_motion=mean_motion,
            )
        )

    return candidates


def score_track_match(
    candidate: BlobCandidate,
    track: MotionTrack,
    frame_size: Tuple[int, int],
) -> float:
    """Score how well a candidate matches an existing persistent track."""
    iou = bbox_iou(candidate.bbox, track.bbox)
    frame_width, frame_height = frame_size
    max_distance = max(1.0, np.hypot(frame_width, frame_height) * MAX_TRACK_DISTANCE_FACTOR)
    distance = np.hypot(
        candidate.centroid[0] - track.centroid[0],
        candidate.centroid[1] - track.centroid[1],
    )
    distance_score = max(0.0, 1.0 - (distance / max_distance))

    shape_delta = abs(candidate.compactness - track.compactness)
    shape_score = max(0.0, 1.0 - shape_delta)

    motion_ratio = 0.0
    if max(candidate.mean_motion, track.mean_motion) > 1e-6:
        motion_ratio = min(candidate.mean_motion, track.mean_motion) / max(
            candidate.mean_motion,
            track.mean_motion,
        )

    persistence_score = min(track.hits / max(TRACK_MIN_HITS, 1), 1.0)
    return (
        (iou * TRACK_IOU_WEIGHT)
        + (distance_score * TRACK_DISTANCE_WEIGHT)
        + (shape_score * TRACK_SHAPE_WEIGHT)
        + (motion_ratio * TRACK_MOTION_WEIGHT)
        + (persistence_score * TRACK_PERSISTENCE_WEIGHT)
    )


def update_motion_tracks(
    tracks: list[MotionTrack],
    candidates: list[BlobCandidate],
    frame_size: Tuple[int, int],
    next_track_id: int,
) -> Tuple[list[MotionTrack], int]:
    """Update persistent tracks from this frame's candidates."""
    updated_tracks = [MotionTrack(**track.__dict__) for track in tracks]
    unmatched_tracks = set(range(len(updated_tracks)))
    unmatched_candidates = set(range(len(candidates)))
    matches: list[Tuple[float, int, int]] = []

    for track_index, track in enumerate(updated_tracks):
        for candidate_index, candidate in enumerate(candidates):
            score = score_track_match(candidate, track, frame_size)
            if score > 0.0:
                matches.append((score, track_index, candidate_index))

    matches.sort(reverse=True, key=lambda item: item[0])
    for score, track_index, candidate_index in matches:
        if track_index not in unmatched_tracks or candidate_index not in unmatched_candidates:
            continue
        if score < 2.0:
            continue

        candidate = candidates[candidate_index]
        track = updated_tracks[track_index]
        track.bbox = candidate.bbox
        track.centroid = candidate.centroid
        track.area = candidate.area
        track.compactness = candidate.compactness
        track.mean_motion = candidate.mean_motion
        track.hits += 1
        track.misses = 0
        track.age += 1

        unmatched_tracks.remove(track_index)
        unmatched_candidates.remove(candidate_index)

    for track_index in unmatched_tracks:
        updated_tracks[track_index].misses += 1
        updated_tracks[track_index].age += 1

    surviving_tracks = [
        track for track in updated_tracks if track.misses <= TRACK_MAX_MISSES
    ]

    for candidate_index in unmatched_candidates:
        candidate = candidates[candidate_index]
        surviving_tracks.append(
            MotionTrack(
                track_id=next_track_id,
                bbox=candidate.bbox,
                centroid=candidate.centroid,
                area=candidate.area,
                compactness=candidate.compactness,
                mean_motion=candidate.mean_motion,
            )
        )
        next_track_id += 1

    return surviving_tracks, next_track_id


def select_output_mask(
    tracks: list[MotionTrack],
    candidates: list[BlobCandidate],
    keep_blobs: int,
) -> np.ndarray | None:
    """Select masks from the highest-confidence persistent tracks."""
    if not tracks or not candidates:
        return None

    candidate_by_bbox = {candidate.bbox: candidate for candidate in candidates}
    stable_tracks = [track for track in tracks if track.hits >= TRACK_MIN_HITS and track.misses == 0]
    if not stable_tracks:
        return None

    stable_tracks.sort(
        key=lambda track: (
            track.confidence,
            track.mean_motion,
            track.area,
        ),
        reverse=True,
    )

    selected_mask: np.ndarray | None = None
    selected_count = 0
    for track in stable_tracks:
        candidate = candidate_by_bbox.get(track.bbox)
        if candidate is None:
            continue
        if selected_mask is None:
            selected_mask = np.zeros_like(candidate.mask)
        selected_mask = cv2.bitwise_or(selected_mask, candidate.mask)
        selected_count += 1
        if selected_count >= keep_blobs:
            break

    return selected_mask


def build_motion_response(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """Compute a coherence-weighted motion response from dense optical flow."""
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude, _ = cv2.cartToPolar(flow_x, flow_y)

    mean_flow_x = cv2.boxFilter(flow_x, ddepth=-1, ksize=(COHERENCE_WINDOW_SIZE, COHERENCE_WINDOW_SIZE))
    mean_flow_y = cv2.boxFilter(flow_y, ddepth=-1, ksize=(COHERENCE_WINDOW_SIZE, COHERENCE_WINDOW_SIZE))
    mean_magnitude = cv2.boxFilter(
        magnitude,
        ddepth=-1,
        ksize=(COHERENCE_WINDOW_SIZE, COHERENCE_WINDOW_SIZE),
    )
    coherent_magnitude = cv2.magnitude(mean_flow_x, mean_flow_y)
    coherence = coherent_magnitude / (mean_magnitude + 1e-6)
    coherence = np.clip(coherence, 0.0, 1.0)

    return magnitude * (coherence ** 2)


def finalize_mask(
    magnitude: np.ndarray,
    threshold: float,
    kernel: np.ndarray,
    keep_blobs: int,
    min_area: int,
    previous_mask: np.ndarray | None,
) -> np.ndarray:
    """Threshold, denoise, and keep the most stable motion regions."""
    mask = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return filter_mask_components(mask, keep_blobs, min_area, previous_mask)


def create_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a colored motion mask onto the original frame."""
    overlay_color = np.zeros_like(frame)
    overlay_color[..., 1] = mask
    return cv2.addWeighted(frame, 1.0, overlay_color, OVERLAY_ALPHA, 0.0)


def process_video(
    input_path: Path,
    out_dir: Path,
    threshold: float,
    downscale: float,
    fps_override: float | None,
    stabilize: bool,
    keep_blobs: int,
    min_area: int,
    ema: float,
    roi: Roi | None,
) -> Tuple[Path, Path]:
    """Read the input video, compute frame-to-frame motion masks, and write outputs."""
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

    first_frame = resize_frame(first_frame, downscale)
    frame_height, frame_width = first_frame.shape[:2]
    roi = validate_roi(roi, (frame_width, frame_height))

    input_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps_override if fps_override is not None else input_fps
    if output_fps is None or output_fps <= 0:
        output_fps = 30.0

    frame_size = (frame_width, frame_height)
    mask_writer = create_writer(mask_path, output_fps, frame_size)
    overlay_writer = create_writer(overlay_path, output_fps, frame_size)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    prev_gray = prepare_gray_for_flow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))
    ema_magnitude: np.ndarray | None = None
    previous_motion_mask: np.ndarray | None = None
    tracks: list[MotionTrack] = []
    next_track_id = 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            frame = resize_frame(frame, downscale)
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

            curr_gray = prepare_gray_for_flow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            aligned_gray = curr_gray
            if stabilize:
                warp_matrix = estimate_ecc_warp(prev_gray, curr_gray)
                aligned_gray = warp_frame_to_previous(curr_gray, warp_matrix)

            if roi is None:
                prev_flow_gray = prev_gray
                curr_flow_gray = aligned_gray
            else:
                x, y, w, h = roi
                prev_flow_gray = prev_gray[y : y + h, x : x + w]
                curr_flow_gray = aligned_gray[y : y + h, x : x + w]

            magnitude = build_motion_response(prev_flow_gray, curr_flow_gray)
            if ema > 0.0:
                if ema_magnitude is None or ema_magnitude.shape != magnitude.shape:
                    ema_magnitude = magnitude.copy()
                else:
                    ema_magnitude = (ema * magnitude) + ((1.0 - ema) * ema_magnitude)
                magnitude = ema_magnitude

            previous_roi_mask: np.ndarray | None = previous_motion_mask
            if roi is not None and previous_motion_mask is not None:
                previous_roi_mask = previous_motion_mask[y : y + h, x : x + w]

            candidate_mask = finalize_mask(
                magnitude,
                threshold,
                kernel,
                keep_blobs,
                min_area,
                previous_roi_mask,
            )
            candidates = extract_blob_candidates(candidate_mask, magnitude, min_area)
            tracks, next_track_id = update_motion_tracks(
                tracks,
                candidates,
                candidate_mask.shape[1::-1],
                next_track_id,
            )
            tracked_mask = select_output_mask(tracks, candidates, keep_blobs)
            if tracked_mask is None:
                motion_mask = np.zeros_like(candidate_mask)
            else:
                motion_mask = tracked_mask

            if roi is None:
                mask = motion_mask
            else:
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                mask[y : y + h, x : x + w] = motion_mask

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = create_overlay(frame, mask)

            mask_writer.write(mask_bgr)
            overlay_writer.write(overlay)
            previous_motion_mask = mask
            prev_gray = curr_gray
    finally:
        capture.release()
        mask_writer.release()
        overlay_writer.release()

    return mask_path, overlay_path


def main() -> int:
    """Entry point for the CLI."""
    try:
        args = parse_args()
        input_path, out_dir, roi = validate_args(args)
        mask_path, overlay_path = process_video(
            input_path=input_path,
            out_dir=out_dir,
            threshold=args.threshold,
            downscale=args.downscale,
            fps_override=args.fps,
            stabilize=not args.no_stabilize,
            keep_blobs=args.keep_blobs,
            min_area=args.min_area,
            ema=args.ema,
            roi=roi,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Mask video written to: {mask_path}")
    print(f"Overlay video written to: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
