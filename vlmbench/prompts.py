"""Prompt builder for clinically significant prostate cancer (csPCa).

Every string produced here is checked byte for byte against the prompts the
original study runners sent (``tests/test_prompts.py``), so wording,
punctuation and whitespace must not be edited casually.

Output formats
--------------
``label``        constrained single-token answer (local models only)
``prob``         verbalised probability, standard wording
``prob_strict``  verbalised probability, wording that forbids any explanation
``json``         label + probabilities as strict JSON (API models)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

FORMATS = ("label", "prob", "prob_strict", "json")


class MissingClinicalValue(ValueError):
    """A clinical measure required by the input type is missing for this case."""


def _json_instruction(label_key: str) -> str:
    return (
        "Reply with strict JSON only, with no markdown or explanation: "
        f'{{"{label_key}": 0 or 1, "prob_0": number between 0 and 1, '
        '"prob_1": number between 0 and 1}. '
        f'The probabilities must sum to 1. Use "{label_key}": 1 when '
        f'prob_1 >= prob_0, otherwise use "{label_key}": 0.'
    )


LABEL_INSTRUCTION = 'Reply with exactly one character: "1" or "0".'


def _binary_instruction(fmt: str, positive_meaning: str, label_key: str) -> str:
    if fmt == "label":
        return LABEL_INSTRUCTION
    if fmt == "prob":
        return (
            "Instead of a single label, output your estimated probability that the "
            f"correct label is 1 ({positive_meaning}), as a single decimal number "
            "between 0.00 and 1.00 "
            "(0.00 = label 0 for certain, 1.00 = label 1 for certain). "
            'Reply with ONLY the number, e.g. "0.73".'
        )
    if fmt == "prob_strict":
        return (
            "Instead of a single label, output your estimated probability that "
            f"the correct label is 1 ({positive_meaning}), as a single decimal number "
            "between 0.00 and 1.00 "
            "(0.00 = label 0 for certain, 1.00 = label 1 for certain).\n"
            'Output only that number, e.g. "0.73", with no other text, no '
            "analysis, and no explanation."
        )
    if fmt == "json":
        return _json_instruction(label_key)
    raise ValueError(f"unknown prompt format {fmt!r}; expected one of {FORMATS}")


# --------------------------------------------------------------------- prostate
PROSTATE_MODALITIES = ("t2", "dwi", "adc")
PROSTATE_DISPLAY = {
    "t2": "T2-weighted (T2W)",
    "dwi": "DWI (high b-value)",
    "adc": "ADC map",
}
# input type -> clinical measures it supplies
PROSTATE_INPUT_TYPES = {
    "image_only": (),
    "image_psa": ("psa",),
    "image_volume": ("prostate_volume",),
    "image_psa_volume": ("psa", "prostate_volume"),
}
_PROSTATE_UNAVAILABLE = {
    "image_only": "No PSA, prostate volume, PSA density, biopsy results, or other "
                  "clinical information is available.",
    "image_psa": "No prostate volume, PSA density, biopsy results, or other clinical "
                 "information is available.",
    "image_volume": "No PSA, PSA density, biopsy results, or other clinical "
                    "information is available.",
    "image_psa_volume": "No PSA density, biopsy results, or other clinical "
                        "information is available.",
}
_PROSTATE_POSITIVE = "clinically significant prostate cancer is present"


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def prostate_mri_text(modalities: Sequence[str]) -> str:
    names = [PROSTATE_DISPLAY[m] for m in modalities]
    if len(modalities) == 1:
        return f"One prostate MRI sequence is provided: {names[0]}."
    return (
        "Multiple prostate MRI sequences from the same case are provided "
        f"in this order: {', '.join(names)}."
    )


def prostate_prompt(input_type: str, modalities: Sequence[str], psa=None,
                    prostate_volume=None, fmt: str = "prob") -> str:
    """csPCa prompt for one modality combination and clinical-input setting."""
    needs = PROSTATE_INPUT_TYPES[input_type]
    lines = [f"- Prostate MRI: {prostate_mri_text(modalities)}"]
    if "psa" in needs:
        value = _as_float(psa)
        if value is None:
            raise MissingClinicalValue("missing_psa")
        lines.append(f"- Serum PSA: {value:.2f} ng/mL")
    if "prostate_volume" in needs:
        value = _as_float(prostate_volume)
        if value is None:
            raise MissingClinicalValue("missing_prostate_volume")
        lines.append(f"- Prostate volume: {value:.2f} mL")
    return (
        "You are performing a binary classification task for clinically significant "
        "prostate cancer.\n\n"
        "Available inputs:\n"
        + "\n".join(lines) + "\n\n"
        f"Unavailable inputs must not be assumed. {_PROSTATE_UNAVAILABLE[input_type]}\n\n"
        "Clinically significant prostate cancer means PI-RADS 4-5 or an equivalent "
        "level of suspicion.\n\n"
        "Use label 1 when the available inputs support clinically significant "
        "prostate cancer. Use label 0 when the available inputs do not support "
        "clinically significant prostate cancer, including negative, benign-appearing, "
        "low-suspicion, equivocal, limited, nonspecific, or insufficient evidence.\n\n"
        + _binary_instruction(fmt, _PROSTATE_POSITIVE, "cspca")
    )
