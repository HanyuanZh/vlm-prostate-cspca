"""Data reader.  It yields :class:`Unit` objects: the images and metadata of one request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Unit:
    """One request: the images in order, plus metadata copied to the output row."""

    key: tuple                  # resume key, unique within one output file
    images: list                # Path objects or PIL images (generated placebo images)
    meta: dict = field(default_factory=dict)
    prompt_args: dict = field(default_factory=dict)  # passed to the prompt builder
    error: Optional[str] = None  # set when the unit cannot be run (e.g. missing clinical value)

