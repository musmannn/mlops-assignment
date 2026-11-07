import io

import numpy as np
import pytest
from PIL import Image

from visionguard.preprocess import UnreadableImageError, load_rgb_letterboxed


def encode(img, fmt="PNG"):
    b = io.BytesIO()
    img.save(b, fmt)
    return b.getvalue()


def test_letterbox_keeps_aspect():
    x = load_rgb_letterboxed(encode(Image.new("RGB", (400, 200), (255, 255, 255))), 224)
    assert x.shape == (224, 224, 3) and x.dtype == np.float32
    assert x[0, 112].sum() == 0          # padded rows (top) are black
    assert x[112, 112].min() == 255      # centre is the image


def test_rgba_and_grayscale_become_rgb():
    for mode in ["RGBA", "L", "P"]:
        assert load_rgb_letterboxed(encode(Image.new(mode, (50, 80))), 96).shape == (96, 96, 3)


def test_bytes_and_path_identical(tmp_path):
    data = encode(Image.new("RGB", (300, 120), (10, 120, 200)), "JPEG")
    p = tmp_path / "a.jpg"
    p.write_bytes(data)
    assert np.array_equal(load_rgb_letterboxed(data, 128), load_rgb_letterboxed(p, 128))


def test_truncated_file_raises():
    data = encode(Image.new("RGB", (500, 500), (1, 2, 3)), "JPEG")
    with pytest.raises(UnreadableImageError):
        load_rgb_letterboxed(data[: len(data) // 3], 64)
