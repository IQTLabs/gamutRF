import prometheus_client
import pytest

from gamutrf import scan


def test_scan_init_prom_vars():
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    scan.init_prom_vars()


def test_scan_argument_parser():
    scan.argument_parser()


def test_bad_freq(mocker):
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    mocker.patch(
        "sys.argv",
        [
            "--freq-start=100",
            "--freq-end=99"
        ],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        scan.main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_bad_rollover(mocker):
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    mocker.patch(
        "sys.argv",
        [
            "--freq-start=99",
            "--freq-end=100"
        ],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        scan.main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_bad_freq_end(mocker):
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    mocker.patch(
        "sys.argv",
        [
            "--freq-end=6.1e9"
        ],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        scan.main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_bad_freq_start(mocker):
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    mocker.patch(
        "sys.argv",
        [
            "--freq-start=69e6"
        ],
    )
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        scan.main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


def test_scan_main(mocker):
    prometheus_client.REGISTRY = prometheus_client.CollectorRegistry(auto_describe=True)
    mocker.patch(
        "sys.argv",
        [
            "--updatetimeout=-1"
        ],
    )
    scan.main()
