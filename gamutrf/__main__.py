"""Main entrypoint for GamutRF"""
from gamutrf.compress_dirs import main as compress_dirs_main
from gamutrf.offline import main as offline_main
from gamutrf.scan import main as scan_main
from gamutrf.sigfinder import main as sigfinder_main
from gamutrf.specgram import main as specgram_main
from gamutrf.worker import main as worker_main


def compress_dirs():
    """Entrypoint for compress_dirs"""
    compress_dirs_main()


def offline():
    """Entrypoint for offline"""
    offline_main()


def scan():
    """Entrypoint for scan"""
    scan_main()


def sigfinder():
    """Entrypoint for sigfinder"""
    sigfinder_main()


def specgram():
    """Entrypoint for specgram"""
    specgram_main()


def worker():
    """Entrypoint for worker"""
    worker_main()
