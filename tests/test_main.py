import sys

from gamutrf.__main__ import api
from gamutrf.__main__ import freqxlator
from gamutrf.__main__ import samples2raw
from gamutrf.__main__ import scan
from gamutrf.__main__ import scan2mp4
from gamutrf.__main__ import sigfinder
from gamutrf.__main__ import specgram


sys.argv.append("-h")


def test_main():
    api() 
    freqxlator()
    samples2raw()
    scan()
    scan2mp4()
    sigfinder()
    specgram()
