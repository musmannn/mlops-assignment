"""Image preprocessing shared by training AND serving.

Both the tf.data training pipeline and the Flask/API inference code call
`load_rgb_letterboxed`, so an image is turned into model input in exactly one way.
The model itself contains its own scaling layer (Rescaling), so callers only ever
hand it uint8-range pixels of shape (size, size, 3).

Design choices (explained in the report, Task 2.1):
- Decode with Pillow: it reads JPEG, PNG, WEBP, BMP, GIF (first frame), which
  tf.io.decode_image does not fully cover (no WEBP).
- Apply EXIF orientation so phone photos are upright.
- Letterbox (resize the long side to `size`, pad the short side with black):
  keeps the aspect ratio, so the model never sees stretched faces or buildings.
  Stretching would change exactly the fine texture cues that separate AI
  images from photos.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Refuse absurd images (decompression bombs) but allow large photos.
Image.MAX_IMAGE_PIXELS = 80_000_000

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}


class UnreadableImageError(ValueError):
    """Raised when bytes cannot be decoded as an image."""


def open_image(source: bytes | str | Path) -> Image.Image:
    """Open and fully decode an image from bytes or a path, raising UnreadableImageError."""
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img.load()  # force full decode: truncated files fail here, not later
    except Exception as exc:  # PIL raises many types (OSError, SyntaxError, DecompressionBombError)
        raise UnreadableImageError(f"cannot decode image: {exc}") from exc
    return img


def to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        # Composite transparency onto black, the same colour as the letterbox padding.
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def letterbox(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def load_rgb_letterboxed(source: bytes | str | Path, size: int) -> np.ndarray:
    """Return a float32 array (size, size, 3) with values in [0, 255]."""
    img = letterbox(to_rgb(open_image(source)), size)
    return np.asarray(img, dtype=np.float32)
