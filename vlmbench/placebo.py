"""Placebo images and the placebo condition grid.

Each placebo request holds the real image(s) of an image-only configuration plus
uninformative images, and reuses that configuration's prompt verbatim: the text
never mentions the extra images.  Three arms differ only in the added pixels:

``dup``    the real image repeated
``blank``  uniform mid-grey, RGB (128, 128, 128)
``noise``  independent uniform RGB noise, seeded from the MD5 of a per-case string,
           so every model receives the same noise image for a given case

Generated images have the pixel size of the real image they accompany and are
sent losslessly (PNG).  ``tests/test_placebo.py`` checks the PNG bytes against the
images used in the study.

Grid: each single MRI sequence + 2 placebo images (dup, blank, noise) and each
sequence pair + 1 placebo image (blank, noise): 15 conditions x 1,500 cases per model.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .data import Unit

BLANK_RGB = (128, 128, 128)


def blank_image(size):
    return Image.new("RGB", size, BLANK_RGB)


def noise_image(size, seed_text: str):
    seed = int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    width, height = size
    return Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8"), "RGB")


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass(frozen=True)
class Placeholder:
    """A placebo image, rendered only when a request is actually sent."""

    kind: str                   # "blank" or "noise"
    size: tuple                 # (width, height)
    seed: Optional[str] = None  # noise only

    def render(self):
        return blank_image(self.size) if self.kind == "blank" else noise_image(self.size, self.seed)


def _size(path):
    with Image.open(path) as im:
        return im.size


def _counts(n_real, n_dup, n_blank, n_noise):
    return {"n_real_modalities": n_real, "n_dup": n_dup, "n_blank": n_blank, "n_noise": n_noise}


def prostate_units(data, case_filter=None):
    """``data`` is a :class:`vlmbench.data.prostate.ProstateData`."""
    for fname in data.files(["t2", "dwi", "adc"]):
        case_key = fname.rsplit(".", 1)[0]
        if case_filter is not None and not case_filter(case_key):
            continue
        paths = {m: data.root / m / fname for m in ("t2", "dwi", "adc")}
        truth = data.truth(case_key)
        for base in ("t2", "dwi", "adc"):
            real, size = paths[base], _size(paths[base])
            blank, noise = Placeholder("blank", size), Placeholder("noise", size, f"{case_key}:{base}")
            for cond, arm, images, counts in (
                (f"{base}_dup_3", "dup", [real, real, real], _counts(1, 2, 0, 0)),
                (f"{base}_blank_3", "blank", [real, blank, blank], _counts(1, 0, 2, 0)),
                (f"{base}_noise_3", "noise", [real, noise, noise], _counts(1, 0, 0, 2)),
            ):
                yield Unit(key=(cond, case_key), images=images,
                           meta={"condition": cond, "arm": arm, "base_modality": base,
                                 "modality_combo": base, "num_images": 3, **counts,
                                 "case_key": case_key, "true_binary": truth},
                           prompt_args={"input_type": "image_only", "modalities": [base]})
        for first, second in (("t2", "dwi"), ("t2", "adc"), ("dwi", "adc")):
            size = _size(paths[first])
            blank = Placeholder("blank", size)
            noise = Placeholder("noise", size, f"{case_key}:{first}_{second}")
            real = [paths[first], paths[second]]
            for cond, arm, images, counts in (
                (f"{first}_{second}_blank_1", "blank", real + [blank], _counts(2, 0, 1, 0)),
                (f"{first}_{second}_noise_1", "noise", real + [noise], _counts(2, 0, 0, 1)),
            ):
                yield Unit(key=(cond, case_key), images=images,
                           meta={"condition": cond, "arm": arm, "base_modality": first,
                                 "modality_combo": f"{first}+{second}", "num_images": 3,
                                 **counts, "case_key": case_key, "true_binary": truth},
                           prompt_args={"input_type": "image_only",
                                        "modalities": [first, second]})
