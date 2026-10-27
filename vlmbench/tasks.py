"""What each experiment runs: its units, prompts, output columns and how a reply is read.

Shared by the local runner, the synchronous API runner and the Batch API path,
so all three send the same requests and write the same columns.
"""

from __future__ import annotations

from .data.prostate import INPUT_TYPES, ProstateData
from .parsing import parse_binary_json, parse_binary_prob
from .placebo import prostate_units
from .prompts import prostate_prompt

DATASET = "prostate"   # value of the ``dataset`` output column
LABEL_KEY = "cspca"

_BINARY_RESULT = ["true_binary", "prob_verbalized", "label_pred"]
_PLACEBO_DESIGN = ["condition", "arm", "base_modality", "modality_combo", "num_images",
                   "n_real_modalities", "n_dup", "n_blank", "n_noise"]

COLUMNS = {
    "main": ["input_type", "modality_combo", "num_modalities", "case_key", "psa",
             "prostate_volume", *_BINARY_RESULT, "prob_verbalized_raw",
             "label_logit", "logit_prob_1", "label_logit_raw"],
    "placebo": [*_PLACEBO_DESIGN, "case_key", *_BINARY_RESULT, "prob_raw"],
}
# columns that identify a unit within one output file (used to resume)
KEY_COLUMNS = {
    "main": ("input_type", "modality_combo", "case_key"),
    "placebo": ("condition", "case_key"),
}


def columns(experiment):
    return ["experiment", "dataset", "model", *COLUMNS[experiment], "error"]


def units(experiment, data_root, types=None):
    """All units of one experiment, in a fixed order."""
    data = ProstateData(data_root)
    if experiment == "placebo":
        yield from prostate_units(data)
    else:
        for t in (types or INPUT_TYPES):
            yield from data.units(t)


def prompt(unit, recipe, label=False):
    """Prompt for ``unit``; ``label=True`` gives the constrained-label variant.

    Raises :class:`vlmbench.prompts.MissingClinicalValue` when the unit lacks a
    clinical measure its input type needs.
    """
    fmt = "label" if label else recipe.prompt_format
    return prostate_prompt(**unit.prompt_args, fmt=fmt)


def interpret(recipe, raw):
    """Result columns for the verbalised reply ``raw``."""
    if recipe.prompt_format == "json":
        p1 = parse_binary_json(raw, LABEL_KEY)[2]
    else:
        p1 = parse_binary_prob(raw)
    out = {"prob_verbalized": p1, "label_pred": None if p1 is None else int(p1 >= 0.5)}
    if p1 is None:
        out["error"] = "unparseable_reply"
    return out


def raw_column(experiment):
    return "prob_verbalized_raw" if experiment == "main" else "prob_raw"


def base_row(unit, experiment, model_key):
    row = {"experiment": experiment, "dataset": DATASET, "model": model_key}
    row.update(unit.meta)
    return row
