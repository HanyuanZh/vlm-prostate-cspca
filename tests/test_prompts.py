"""The prompts must match, byte for byte, what the original study runners sent.

``fixtures/golden.json`` holds the MD5 of every prompt rendered from the original
scripts (all four local model families and both API providers were rendered
separately and found identical before the fixture was written).
"""
import hashlib
import json
from pathlib import Path

import pytest

from vlmbench.prompts import prostate_prompt

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text())
PROMPT_KEYS = sorted(k for k in GOLDEN if k.startswith("prostate|"))


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def render(key):
    _, input_type, combo, psa, vol, fmt = key.split("|")
    return prostate_prompt(input_type, combo.split("+"), psa=float(psa),
                           prostate_volume=float(vol), fmt=fmt)


def test_fixture_is_complete():
    # 4 input types x 7 combos x 4 clinical points x 4 formats
    assert len(PROMPT_KEYS) == 448


@pytest.mark.parametrize("key", PROMPT_KEYS)
def test_prompt_matches_original(key):
    assert md5(render(key)) == GOLDEN[key]


def test_placebo_prompts_match_submitted_requests():
    """The placebo arms reuse the image-only JSON prompt verbatim."""
    rendered = {md5(prostate_prompt("image_only", c.split("+"), fmt="json"))
                for c in ("t2", "dwi", "adc", "t2+dwi", "t2+adc", "dwi+adc")}
    for key, submitted in GOLDEN["submitted_placebo_prompts"].items():
        assert set(submitted) <= rendered, key
