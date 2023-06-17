from importlib import metadata

try:
    __version__ = metadata.version("gamutrf")
except metadata.PackageNotFoundError:
    __version__ = "UNKNOWN"
