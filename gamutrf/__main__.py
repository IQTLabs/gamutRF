"""Main entrypoint for GamutRF"""
from gamutrf.worker import main as worker_main
from gamutrf.compress_dirs import main as compress_dirs_main
from gamutrf.freqxlator import main as freqxlator_main
from gamutrf.samples2raw import main as samples2raw_main
from gamutrf.scan import main as scan_main
from gamutrf.sigfinder import main as sigfinder_main
from gamutrf.specgram import main as specgram_main
from gamutrf.waterfall import main as waterfall_main


def worker():
    """Entrypoint for worker"""
    worker_main()


def compress_dirs():
    """Entrypoint for compress_dirs"""
    compress_dirs_main()


def freqxlator():
    """Entrypoint for freqxlator"""
    freqxlator_main()


def samples2raw():
    """Entrypoint for samples2raw"""
    samples2raw_main()


def scan():
    """Entrypoint for scan"""
    scan_main()


def sigfinder():
    """Entrypoint for sigfinder"""
    sigfinder_main()


def specgram():
    """Entrypoint for specgram"""
    specgram_main()


def waterfall():
    """Entrypoint for waterfall"""
    waterfall_main()
