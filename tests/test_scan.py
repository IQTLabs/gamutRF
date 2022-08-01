import sys

from gamutrf import scan


def test_scan_init_prom_vars():
    scan.init_prom_vars()


def test_scan_argument_parser():
    scan.argument_parser()


def test_bad_freq():
    sys.argv.append(["--freq-start", "100", "--freq-end", "99"])
    scan.main()


def test_bad_rollover():
    sys.argv.append(["--freq-start", "99", "--freq-end", "100"])
    scan.main()


def test_bad_freq_end():
    sys.argv.append(["--freq-end", "100000000000"])
    scan.main()


def test_bad_freq_start():
    sys.argv.append(["--freq-start", "10000"])
    scan.main()


def test_scan_main():
    sys.argv.append(["--updatetimeout", "-1"])
    scan.main()
