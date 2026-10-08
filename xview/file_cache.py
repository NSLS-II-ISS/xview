"""Bounded cache for read-only previews of binned scan files."""

from collections import OrderedDict
from pathlib import Path


class BinnedFileCache:
    """Reuse parsed frames until a file changes, with entry and byte limits.

    The loader returns ``(DataFrame, header)``. Callers must treat the returned
    objects as read-only; project imports that modify data should load afresh.
    """

    def __init__(self, loader, max_entries=32, max_bytes=64 * 1024 * 1024):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Cache limits must be positive")
        self.loader = loader
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def clear(self):
        self._entries.clear()
        self._bytes = 0

    def load(self, filename):
        path = Path(filename).resolve()
        signature = self._signature(path)
        cached = self._entries.pop(path, None)
        if cached is not None:
            old_signature, result, size = cached
            self._bytes -= size
            if old_signature == signature:
                self._entries[path] = cached
                self._bytes += size
                return result

        result = self.loader(str(path))
        frame, header = result
        size = int(frame.memory_usage(index=True, deep=True).sum()) + len(header.encode("utf-8"))
        # Never retain a partial read of a scan still being written.
        if size <= self.max_bytes and self._signature(path) == signature:
            while self._entries and (
                len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes
            ):
                _, (_, _, evicted_size) = self._entries.popitem(last=False)
                self._bytes -= evicted_size
            self._entries[path] = signature, result, size
            self._bytes += size
        return result
