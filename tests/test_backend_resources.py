"""Use real Streamlit caching to guard backend-only deployments; no model/API."""
from concurrent.futures import ThreadPoolExecutor
import sys
from unittest.mock import patch

import pytest

from backend_resources import load_core_resources


@pytest.fixture(autouse=True)
def clean_resource_cache():
    load_core_resources.clear()
    yield
    load_core_resources.clear()


def source(version):
    return f'''
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
_resources = None

def load_metadata():
    return {{"photo": {{"text": {version!r}}}}}

def parse_query(query):
    return {version!r}, query

def get_search_resources():
    global _resources
    if _resources is None:
        _resources = {{"index_text": load_metadata()["photo"]["text"]}}
    return _resources
'''.encode()


def test_unchanged_source_reuses_backend_and_lazy_index(tmp_path):
    path = str(tmp_path / "AdaphotoRet_run.py")
    first = load_core_resources(source("old"), path)
    second = load_core_resources(source("old"), path)
    assert first is second
    assert first["backend"].get_search_resources() is second["backend"].get_search_resources()
    assert first["backend"].BASE_DIR == tmp_path


def test_changed_source_replaces_stale_module_and_index_without_mutating_old(tmp_path):
    path = str(tmp_path / "AdaphotoRet_run.py")
    old = load_core_resources(source("old"), path)
    old_index = old["backend"].get_search_resources()
    # A normal import would return this stale object even after a cache miss.
    with patch.dict(sys.modules, {"AdaphotoRet_run": old["backend"]}):
        new = load_core_resources(source("new"), path)
        assert sys.modules["AdaphotoRet_run"] is old["backend"]
    assert new is not old
    assert new["backend"].parse_query("same clue") == ("new", "same clue")
    assert new["metadata"]["photo"]["text"] == "new"
    assert new["backend"].get_search_resources()["index_text"] == "new"
    assert old_index["index_text"] == "old"
    assert old["backend"].parse_query("same clue") == ("old", "same clue")


def test_source_content_not_mtime_and_no_re_read_during_load(tmp_path):
    path = tmp_path / "AdaphotoRet_run.py"
    path.write_bytes(source("old"))
    snapshot = path.read_bytes()
    before = path.stat()
    path.write_bytes(source("new"))
    import os
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    old = load_core_resources(snapshot, str(path))
    new = load_core_resources(path.read_bytes(), str(path))
    assert old["metadata"]["photo"]["text"] == "old"
    assert new["metadata"]["photo"]["text"] == "new"


def test_concurrent_visitors_share_only_completed_same_version_module(tmp_path):
    path = str(tmp_path / "AdaphotoRet_run.py")
    with ThreadPoolExecutor(max_workers=4) as workers:
        cores = list(workers.map(lambda _: load_core_resources(source("new"), path), range(12)))
    assert all(core is cores[0] for core in cores)
    assert all(core["backend"].parse_query("query") == ("new", "query") for core in cores)


def test_failed_version_does_not_corrupt_held_backend(tmp_path):
    path = str(tmp_path / "AdaphotoRet_run.py")
    old = load_core_resources(source("old"), path)
    with pytest.raises(RuntimeError, match="fixture failure"):
        load_core_resources(b"raise RuntimeError('fixture failure')", path)
    assert old["backend"].parse_query("query") == ("old", "query")
    assert load_core_resources(source("new"), path)["metadata"]["photo"]["text"] == "new"
