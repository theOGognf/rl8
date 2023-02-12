"""Top-level package interface."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("finagg")
except PackageNotFoundError:
    pass
