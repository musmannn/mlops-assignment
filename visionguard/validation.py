"""Upload validation shared by the web app and the model API (Task 4.2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from visionguard.preprocess import ALLOWED_EXTENSIONS, UnreadableImageError, open_image

REAL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "MPO"}  # what Pillow must detect from the bytes


class UploadError(ValueError):
    def __init__(self, message: str, status: int = 400, title: str = "Invalid upload"):
        super().__init__(message)
        self.status, self.title = status, title


@dataclass
class ValidUpload:
    filename: str
    data: bytes
    sha256: str
    width: int
    height: int
    fmt: str


def validate_upload(file: FileStorage | None) -> ValidUpload:
    if file is None or not file.filename:
        raise UploadError("Please choose an image file to upload.", 400, "No file selected")

    # secure_filename strips paths and odd characters ("../../etc/passwd" -> "etc_passwd").
    # The name is only displayed and logged; the file is never written to disk.
    filename = secure_filename(file.filename)[:200] or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(
            f"'.{ext or '?'}' files are not accepted. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            415, "Unsupported file type")

    data = file.read()  # held in memory only (size already capped by MAX_CONTENT_LENGTH)
    if not data:
        raise UploadError("The uploaded file is empty.", 400, "Empty file")

    # The extension can lie; decode the actual bytes.
    try:
        img = open_image(data)
    except UnreadableImageError:
        raise UploadError("This file has an image extension but could not be decoded as an image. "
                          "It may be corrupt or not really an image.", 400, "Unreadable image") from None
    if img.format not in REAL_FORMATS:
        raise UploadError(f"Detected format {img.format} is not supported.", 415, "Unsupported file type")
    return ValidUpload(filename, data, hashlib.sha256(data).hexdigest(), img.width, img.height, img.format)
