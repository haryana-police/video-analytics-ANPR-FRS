"""Indian license plate format validation.

Indian plates follow the BH-series national format (Bharat series, mandatory
since 2021) and state-specific variants. This module provides a permissive
validator that catches obviously wrong OCR outputs while still accepting
the variety of formats seen on the road.

Examples that VALIDATE:
    BH12AB1234       (BH series, the current national standard)
    DL3CAB1234       (state code Delhi, 1-2 digit district, 1-3 alpha series,
                      1-4 digit number — the classic pre-BH format)
    MH12AB3456
    TN01AB0001
    KA51MA1234       (KA-51 Mangalore RTO, two-alpha series MA)
    HR26DK8330       (HR-26 Gurugaon)
    1234             (rejected — too short)
    ABCDEFGH         (rejected — all letters)

Format grammar (loose):
    ^([A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4})$    length 6-13
        BH series: [A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}    (length 10, fixed)

The `validate_indian_plate(text)` function:
    * upper-cases & strips whitespace
    * rejects anything shorter than 6 or longer than 13 alphanumerics
    * requires at least 2 letters AND at least 2 digits (rejects "ABCDEFG")
    * matches the BH-series grammar OR the loose state grammar
    * returns (ok: bool, normalized_text: str)
"""

from __future__ import annotations

import re

# BH-series (Bharat): state code (2 alpha) + year (2 digit) + letters (2 alpha) + number (4 digit)
_BH_SERIES_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$")

# State-series (legacy): state code (2 alpha) + district (1-2 digit) +
# series (1-3 alpha) + number (1-4 digit), length 6-13.
_STATE_SERIES_RE = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"
)

# State code list (RTO codes). Used as a sanity check — at least the first
# 2 letters must be a known state code (or BH for the new national format).
_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

_BH_CODE = "BH"


def _normalize(text: str) -> str:
    return (text or "").upper().strip().replace(" ", "").replace("-", "")


def validate_indian_plate(text: str) -> tuple[bool, str]:
    """Return (ok, normalized_text). ok=True iff the text plausibly looks
    like an Indian license plate."""
    norm = _normalize(text)
    # strip non-alphanumeric noise
    alnum = "".join(c for c in norm if c.isalnum())
    if len(alnum) < 6 or len(alnum) > 13:
        return False, alnum
    # must contain at least 2 letters AND 2 digits (avoids "ABCDEF" and "123456")
    if sum(c.isalpha() for c in alnum) < 2:
        return False, alnum
    if sum(c.isdigit() for c in alnum) < 2:
        return False, alnum
    # first 2 chars must be a state code or BH
    if alnum[:2] not in (_STATE_CODES | {_BH_CODE}):
        return False, alnum
    # exact BH-series grammar wins
    if _BH_SERIES_RE.match(alnum):
        return True, alnum
    # loose state-series grammar
    if _STATE_SERIES_RE.match(alnum):
        return True, alnum
    return False, alnum