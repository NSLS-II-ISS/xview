"""Filesystem paths for bundled Qt forms and other package data."""

import atexit
from contextlib import ExitStack
from functools import lru_cache
from importlib.resources import as_file, files


_resource_contexts = ExitStack()
atexit.register(_resource_contexts.close)


@lru_cache(maxsize=None)
def resource_filename(package, name):
    """Keep extracted resources alive for Qt, including in zipped packages."""
    resource = files(package).joinpath(name)
    return str(_resource_contexts.enter_context(as_file(resource)))
