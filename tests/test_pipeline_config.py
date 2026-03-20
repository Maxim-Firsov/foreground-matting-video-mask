from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

try:
    from src.motion_mask_pipeline import PipelineConfig, RuntimeStats, build_run_metadata, mask_coverage_ratio, parse_roi, scale_roi_to_frame, update_runtime_stats, validate_roi
    from src.foreground_mask import resolve_profile, resolve_profile_from_dimensions, validate_args
except ModuleNotFoundError as exc:
    RuntimeStats = None
    PipelineConfig = None
    build_run_metadata = None
    mask_coverage_ratio = None
    parse_roi = None
    scale_roi_to_frame = None
    update_runtime_stats = None
    validate_roi = None
    resolve_profile = None
    resolve_profile_from_dimensions = None
    validate_args = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"pipeline dependencies unavailable: {IMPORT_ERROR}")
class PipelineConfigTests(unittest.TestCase):
    def test_parse_roi_accepts_valid_values(self) -> None:
        self.assertEqual(parse_roi("10,20,300,200"), (10, 20, 300, 200))

    def test_parse_roi_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            parse_roi("10,20,300")

    def test_validate_roi_rejects_out_of_bounds_regions(self) -> None:
        with self.assertRaises(ValueError):
            validate_roi((10, 20, 800, 600), (640, 360))

    def test_scale_roi_to_frame_downscales_coordinates(self) -> None:
        self.assertEqual(scale_roi_to_frame((100, 40, 320, 200), 0.5), (50, 20, 160, 100))

    def test_mask_coverage_ratio_counts_foreground_pixels(self) -> None:
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        self.assertEqual(mask_coverage_ratio(mask), 0.5)

    @patch("src.foreground_mask.inspect_video", return_value=(4096, 2160))
    def test_auto_profile_downscales_large_clip(self, _inspect_video) -> None:
        args = type("Args", (), {"profile": "auto", "downscale": None, "no_stabilize": False})()
        downscale, stabilize = resolve_profile(args, Path("demo.mp4"))
        self.assertEqual(downscale, 0.25)
        self.assertFalse(stabilize)

    @patch("src.foreground_mask.inspect_video", return_value=(1280, 720))
    def test_quality_profile_preserves_resolution(self, _inspect_video) -> None:
        args = type("Args", (), {"profile": "quality", "downscale": None, "no_stabilize": False})()
        downscale, stabilize = resolve_profile(args, Path("demo.mp4"))
        self.assertEqual(downscale, 1.0)
        self.assertTrue(stabilize)

    def test_auto_profile_uses_mid_resolution_defaults_at_full_hd_boundary(self) -> None:
        args = type("Args", (), {"profile": "auto", "downscale": None, "no_stabilize": False})()

        downscale, stabilize = resolve_profile_from_dimensions(args, 1920, 1080)

        self.assertEqual(downscale, 0.5)
        self.assertFalse(stabilize)

    def test_validate_args_rejects_non_positive_max_frames(self) -> None:
        args = type(
            "Args",
            (),
            {
                "input": __file__,
                "out_dir": "outputs",
                "threshold": 1.5,
                "downscale": 1.0,
                "fps": None,
                "no_stabilize": False,
                "keep_blobs": 1,
                "min_area": 500,
                "ema": 0.0,
                "roi": None,
                "profile": "quality",
                "max_frames": 0,
            },
        )()
        with self.assertRaisesRegex(ValueError, "--max-frames must be >= 1"):
            validate_args(args)

    @patch("src.foreground_mask.inspect_video", return_value=(1280, 720))
    def test_validate_args_passes_max_frames_into_pipeline_config(self, _inspect_video) -> None:
        args = type(
            "Args",
            (),
            {
                "input": __file__,
                "out_dir": "outputs",
                "threshold": 1.5,
                "downscale": None,
                "fps": None,
                "no_stabilize": False,
                "keep_blobs": 1,
                "min_area": 500,
                "ema": 0.0,
                "roi": None,
                "profile": "fast",
                "max_frames": 12,
            },
        )()

        _, _, config = validate_args(args)

        self.assertEqual(config.max_frames, 12)

    @patch("src.foreground_mask.inspect_video", return_value=(4096, 2160))
    def test_validate_args_scales_roi_into_processed_frame_space(self, _inspect_video) -> None:
        args = type(
            "Args",
            (),
            {
                "input": __file__,
                "out_dir": "outputs",
                "threshold": 1.5,
                "downscale": None,
                "fps": None,
                "no_stabilize": False,
                "keep_blobs": 1,
                "min_area": 500,
                "ema": 0.0,
                "roi": "400,200,800,600",
                "profile": "auto",
                "max_frames": None,
            },
        )()

        _, _, config = validate_args(args)

        self.assertEqual(config.downscale, 0.25)
        self.assertEqual(config.roi, (100, 50, 200, 150))

    @patch("src.foreground_mask.inspect_video", return_value=(4096, 2160))
    def test_validate_args_rejects_source_roi_outside_input_frame(self, _inspect_video) -> None:
        args = type(
            "Args",
            (),
            {
                "input": __file__,
                "out_dir": "outputs",
                "threshold": 1.5,
                "downscale": None,
                "fps": None,
                "no_stabilize": False,
                "keep_blobs": 1,
                "min_area": 500,
                "ema": 0.0,
                "roi": "3900,2000,400,300",
                "profile": "auto",
                "max_frames": None,
            },
        )()

        with self.assertRaisesRegex(ValueError, "exceeds frame bounds"):
            validate_args(args)

    def test_update_runtime_stats_tracks_average_and_peak_coverage(self) -> None:
        stats = RuntimeStats()

        update_runtime_stats(stats, 1, np.array([[0, 255], [0, 255]], dtype=np.uint8))
        update_runtime_stats(stats, 2, np.array([[0, 0], [0, 255]], dtype=np.uint8))

        self.assertAlmostEqual(stats.average_mask_coverage, 0.375)
        self.assertAlmostEqual(stats.max_mask_coverage, 0.5)

    def test_build_run_metadata_includes_frame_count_and_early_stop_flag(self) -> None:
        config = PipelineConfig(
            threshold=1.5,
            downscale=0.5,
            fps_override=None,
            stabilize=False,
            keep_blobs=1,
            min_area=500,
            ema=0.2,
            roi=(10, 20, 30, 40),
            max_frames=12,
        )
        stats = RuntimeStats(
            average_mask_coverage=0.25,
            max_mask_coverage=0.5,
            stabilization_attempts=10,
            stabilization_successes=8,
            stabilization_failures=2,
        )

        metadata = build_run_metadata(
            input_path=Path("demo.mp4"),
            input_frame_size=(4096, 2160),
            output_frame_size=(2048, 1080),
            source_frame_count=166,
            processed_frames=12,
            output_fps=25.0,
            config=config,
            runtime_stats=stats,
        )

        self.assertEqual(metadata["source_frame_count"], 166)
        self.assertTrue(metadata["stopped_early"])
        self.assertEqual(metadata["config"]["roi"], (10, 20, 30, 40))
        self.assertEqual(metadata["processing_stats"]["stabilization_successes"], 8)


if __name__ == "__main__":
    unittest.main()
