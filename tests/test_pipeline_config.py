from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from src.motion_mask_pipeline import parse_roi, validate_roi
    from src.foreground_mask import resolve_profile, validate_args
except ModuleNotFoundError as exc:
    parse_roi = None
    validate_roi = None
    resolve_profile = None
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


if __name__ == "__main__":
    unittest.main()
