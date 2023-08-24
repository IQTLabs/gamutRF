import pytest

from gamutrf.scan import argument_parser
from gamutrf.scan import main


def test_scan_argument_parser():
    argument_parser()


def test_bad_freq(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["scan.py", "--freq-start=100", "--freq-end=99"],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_bad_rollover(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["scan.py", "--freq-start=9", "--freq-end=100"],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_bad_freq_end(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["scan.py", "--freq-end=6.1e9"],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_bad_freq_start(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["scan.py", "--freq-start=9e6"],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_scan_main(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["scan.py", "--updatetimeout=-1"],
    )
    # there is no SDR connected in test, so expected to fail at runtime
    with pytest.raises(RuntimeError) as pytest_wrapped_e:
        main()
    assert pytest_wrapped_e.type == RuntimeError
