import pytest
import sys

from gamutrf.__main__ import worker
from gamutrf.__main__ import freqxlator
from gamutrf.__main__ import samples2raw
from gamutrf.__main__ import scan
from gamutrf.__main__ import sigfinder
from gamutrf.__main__ import specgram


sys.argv.append("-h")


def test_main_worker():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        worker()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_main_freqxlator():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        freqxlator()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_main_samples2raw():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        samples2raw()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_main_scan():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        scan()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_main_sigfinder():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        sigfinder()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0


def test_main_specgram():
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        specgram()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0
