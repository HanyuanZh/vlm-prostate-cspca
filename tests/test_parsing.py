"""The parsers must reproduce the probabilities stored in the study's final tables.

``fixtures/parsing_cases.json`` is a stratified sample of (raw reply, stored value)
pairs.  Run against the full tables, the same parsers reproduced the stored
probabilities up to the four-decimal rounding of a few table values, hence the
tolerance below.
"""
import json
from pathlib import Path

import pytest

from vlmbench.parsing import parse_binary_json, parse_binary_prob

CASES = json.loads((Path(__file__).parent / "fixtures" / "parsing_cases.json").read_text())
TOL = 1e-4


def _parse(case):
    if case["kind"] == "binary_local":
        return parse_binary_prob(case["raw"])
    return parse_binary_json(case["raw"], case["label_key"])[2]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["kind"])
def test_parser_reproduces_stored_value(case):
    got, expected = _parse(case), case["expected"]
    if expected is None:
        assert got is None
    else:
        assert got is not None and abs(got - expected) < TOL


def test_spaced_decimal_is_not_read_as_zero():
    # MedGemma-4B occasionally writes "0.  85"; it must parse as 0.85, not 0.
    assert parse_binary_prob("0.  85") == pytest.approx(0.85)


def test_json_probabilities_are_renormalised():
    assert parse_binary_json('{"cspca": 1, "prob_0": 0.2, "prob_1": 0.6}', "cspca") == (
        1, pytest.approx(0.25), pytest.approx(0.75))
