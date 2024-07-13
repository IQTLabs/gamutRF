"""Main entrypoint for GamutRF"""

from gamutrf.compress_dirs import main as compress_dirs_main
from gamutrf.offline import main as offline_main
from gamutrf.scan import main as scan_main


def compress_dirs():
    """Entrypoint for compress_dirs"""
    compress_dirs_main()


def offline():
    """Entrypoint for offline"""
    offline_main()


def scan():
    """Entrypoint for scan"""
    scan_main()
