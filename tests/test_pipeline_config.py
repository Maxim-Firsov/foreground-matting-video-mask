from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from src.motion_mask_pipeline import parse_roi, validate_roi
    from src.foreground_mask import resolve_profile
except ModuleNotFoundError as exc:
    parse_roi = None
    validate_roi = None
    resolve_profile = None
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


if __name__ == "__main__":
    unittest.main()
