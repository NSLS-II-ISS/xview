from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from xview.file_cache import BinnedFileCache


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scan.dat"
    path.write_text("energy i0\n1 2\n3 4\n")
    return path


def reader(path):
    return pd.read_csv(path, sep=r"\s+"), "header"


def test_repeated_preview_reads_once(scan):
    loader = Mock(side_effect=reader)
    cache = BinnedFileCache(loader)
    first = cache.load(scan)
    assert cache.load(str(scan)) is first
    loader.assert_called_once_with(str(scan))


def test_changed_file_is_reloaded(scan):
    cache = BinnedFileCache(reader)
    cache.load(scan)
    scan.write_text("energy i0\n1 200\n")
    assert cache.load(scan)[0].i0.tolist() == [200]


def test_same_size_atomic_replacement_is_reloaded(scan):
    cache = BinnedFileCache(reader)
    cache.load(scan)
    replacement = scan.with_suffix(".new")
    replacement.write_text("energy i0\n1 9\n3 8\n")
    replacement.replace(scan)
    assert cache.load(scan)[0].i0.tolist() == [9, 8]


def test_least_recently_used_entry_is_evicted(scan):
    loader = Mock(side_effect=reader)
    cache = BinnedFileCache(loader, max_entries=2)
    second, third = scan.with_name("second.dat"), scan.with_name("third.dat")
    for path in (second, third):
        path.write_text(scan.read_text())
    cache.load(scan)
    cache.load(second)
    cache.load(scan)
    cache.load(third)
    cache.load(scan)
    assert loader.call_count == 3
    cache.load(second)
    assert loader.call_count == 4


def test_byte_budget_and_clear(scan):
    frame, header = reader(scan)
    size = int(frame.memory_usage(index=True, deep=True).sum()) + len(header.encode())
    loader = Mock(side_effect=reader)
    cache = BinnedFileCache(loader, max_bytes=size)
    second = scan.with_name("second.dat")
    second.write_text(scan.read_text())
    cache.load(scan)
    cache.load(second)
    cache.load(scan)
    assert loader.call_count == 3
    cache.clear()
    cache.load(scan)
    assert loader.call_count == 4


def test_oversized_results_are_not_retained(scan):
    loader = Mock(side_effect=reader)
    cache = BinnedFileCache(loader, max_bytes=1)
    cache.load(scan)
    cache.load(scan)
    assert loader.call_count == 2


def test_failed_read_is_retried(scan):
    loader = Mock(side_effect=[ValueError("incomplete scan"), reader(scan)])
    cache = BinnedFileCache(loader)
    with pytest.raises(ValueError, match="incomplete scan"):
        cache.load(scan)
    assert cache.load(scan)[0].i0.tolist() == [2, 4]


def test_file_modified_during_read_is_not_cached(scan):
    def growing_reader(path):
        result = reader(path)
        with Path(path).open("a") as stream:
            stream.write("5 6\n")
        return result

    loader = Mock(side_effect=growing_reader)
    cache = BinnedFileCache(loader)
    cache.load(scan)
    assert len(cache.load(scan)[0]) == 3
    assert loader.call_count == 2


def test_deleted_file_does_not_return_stale_data(scan):
    cache = BinnedFileCache(reader)
    cache.load(scan)
    scan.unlink()
    with pytest.raises(FileNotFoundError):
        cache.load(scan)
