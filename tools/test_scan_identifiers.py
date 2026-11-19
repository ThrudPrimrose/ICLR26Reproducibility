"""A multi-word allow rule exempts its own words from the person scan, and only there."""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import scan_identifiers as scan  # noqa: E402


def test_credit_phrase_is_exempt_but_the_bare_name_still_flags() -> None:
    scan.ALLOW[:] = [re.compile(r"Adalind Q\. Example", re.IGNORECASE)]
    scan.PEOPLE.clear()
    scan.PEOPLE.update({"adalind"})
    assert scan.person_hits('Copyright (c) 2017 Adalind Q. Example, "Some Book"') == 0
    assert scan.person_hits("Reviewed by Adalind, 2026") == 1
    assert scan.person_hits("Adalind Q. Example and Adalind") == 1
