import sys
from gamutrf import scan2rtlpow


def test_scan2rtlpow_main():
    sys.argv = [sys.argv[0], 'tests/data/verybusy1g1.csv', 'out.csv']
    scan2rtlpow.main()
