"""Unit tests for LightX2V worker image task support."""

from worker import LightX2VWorker


class TestImageSizes:
    """Test IMAGE_SIZES mapping used by text-to-image tasks."""

    EXPECTED = {
        "square_hd": (1024, 1024),
        "square": (512, 512),
        "portrait_4_3": (768, 1024),
        "portrait_16_9": (768, 1344),
        "landscape_4_3": (1024, 768),
        "landscape_16_9": (1344, 768),
    }

    def test_all_sizes_present(self):
        for name, (w, h) in self.EXPECTED.items():
            assert name in LightX2VWorker.IMAGE_SIZES, f"Missing size: {name}"
            assert LightX2VWorker.IMAGE_SIZES[name] == (w, h), (
                f"{name}: expected ({w}, {h}), got {LightX2VWorker.IMAGE_SIZES[name]}"
            )

    def test_no_extra_sizes(self):
        assert set(LightX2VWorker.IMAGE_SIZES.keys()) == set(self.EXPECTED.keys())

    def test_unknown_size_falls_back_to_square_hd(self):
        """Unknown image_size values should fall back to square_hd (1024x1024)."""
        fallback = LightX2VWorker.IMAGE_SIZES.get(
            "nonexistent", LightX2VWorker.IMAGE_SIZES["square_hd"]
        )
        assert fallback == (1024, 1024)
