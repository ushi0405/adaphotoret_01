"""Storage contracts are testable without Streamlit or access to user files."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from photo_library_store import (
    FORMAT, MAX_BYTES, LibraryDataError, empty_library, normalize_library,
    export_library, export_legacy_library, import_library, merge_libraries, load_local_library, save_local_library,
    new_browser_state, consume_browser_event, queue_browser_save, request_browser_reload,
)


def album(name="旅行", path="data/猫.jpg"):
    return {"photo_tags": {path: ["生日"]}, "libraries": {name: {"photos": [path], "tags": ["生日"]}}}


def event(state, kind="loaded", data=None, revision=None, event_id="event", **extra):
    return {"request_id": state["request"].get("id"), "event_id": event_id,
            "kind": kind, "revision": revision, "data": data if data is not None else empty_library(), **extra}


class LibrarySchemaTests(unittest.TestCase):
    def test_round_trip_only_known_fields_and_detached_data(self):
        data = album("<img onerror=alert(1)>")
        output = import_library(export_library(data))
        self.assertEqual(output, data)
        output["libraries"].clear()
        self.assertTrue(data["libraries"])
        self.assertEqual(set(json.loads(export_library(data))), {"format", "version", "data"})

    def test_legacy_aliases_sections_migrate_without_losing_photos(self):
        value = {"libraries": {"旧相册": {"photos": ["missing.jpg"], "tags": ["生日"],
                    "aliases": ["生日", "一岁"], "sections": {"幼年": ["两岁"]}}}}
        original = deepcopy(value)
        restored = normalize_library(value)
        self.assertEqual(restored["libraries"]["旧相册"], {"photos": ["missing.jpg"], "tags": ["生日", "一岁", "两岁"]})
        self.assertEqual(value, original)

    def test_legacy_recovery_preserves_oversized_labels_but_never_unknown_secret_fields(self):
        data = album("x" * 201)
        self.assertEqual(json.loads(export_legacy_library(data)), data)
        with self.assertRaises(LibraryDataError):
            export_legacy_library({"api_key": "never", **data})
        with self.assertRaises(LibraryDataError):
            export_legacy_library({"libraries": {"x": {"api_key": "never"}}})

    def test_rejects_wrong_types_unknown_fields_credentials_and_versions(self):
        invalid = [[], {"api_key": "never save"}, {"conversations": {}}, {"photo_tags": []},
                   {"libraries": {"name": []}}, {"libraries": {"name": {"photos": "x.jpg"}}},
                   {"libraries": {"name": {"images": ["blob"]}}}, {"photo_tags": {"x.jpg": [12]}},
                   {"libraries": {"": {}}}, {"libraries": {"x" * 201: {}}},
                   {"format": FORMAT, "version": True, "data": empty_library()},
                   {"format": FORMAT, "version": 2, "data": empty_library()}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(LibraryDataError):
                import_library(json.dumps(value))
        for raw in (b"\xff", b"{broken", b"x" * (MAX_BYTES + 1), b'{"libraries":{"\\ud800":{}}}'):
            with self.assertRaises(LibraryDataError):
                import_library(raw)

    def test_merge_same_name_is_additive_and_preserves_missing_catalogue_refs(self):
        current, incoming = album(), album(path="unavailable-now.jpg")
        result = merge_libraries(current, incoming)
        self.assertEqual(result["libraries"]["旅行"]["photos"], ["data/猫.jpg", "unavailable-now.jpg"])
        self.assertEqual(current, album())

    def test_oversized_legacy_combined_tags_rejected_without_truncation(self):
        with self.assertRaises(LibraryDataError):
            normalize_library({"libraries": {"old": {"tags": [f"a{i}" for i in range(3000)],
                                                        "aliases": [f"b{i}" for i in range(3000)]}}})

    def test_near_limit_backup_and_local_file_are_self_readable(self):
        value = {"libraries": {"large": {"photos": [f"{i:04d}-" + "x" * 1990 for i in range(1030)]}}}
        normalized = normalize_library(value)
        content = export_library(value)
        self.assertLess(len(content), MAX_BYTES)
        self.assertGreater(len(content), MAX_BYTES * .95)
        self.assertEqual(import_library(content), normalized)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "library.json"
            save_local_library(path, value)
            self.assertEqual(load_local_library(path), normalized)

    def test_local_atomic_failure_preserves_previous_bytes_and_no_temp_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "library.json"
            save_local_library(path, album())
            old = path.read_bytes()
            with patch("photo_library_store.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    save_local_library(path, empty_library())
            self.assertEqual(path.read_bytes(), old)
            self.assertEqual(list(Path(temp).iterdir()), [path])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_corrupt_local_load_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "library.json"
            path.write_bytes(b"{broken")
            with self.assertRaises(LibraryDataError):
                load_local_library(path)
            self.assertEqual(path.read_bytes(), b"{broken")


class BrowserControllerTests(unittest.TestCase):
    def ready(self, data=None):
        state = new_browser_state()
        consume_browser_event(state, event(state, data=data))
        return state

    def test_restore_before_mutate_and_two_sessions_isolated(self):
        state, other = new_browser_state(), new_browser_state()
        self.assertNotEqual(state["request"]["id"], other["request"]["id"])
        with self.assertRaises(LibraryDataError):
            queue_browser_save(state, album())
        restored = consume_browser_event(state, event(state, data=album()))
        self.assertEqual(restored, album())
        self.assertEqual(other["phase"], "loading")

    def test_saved_only_after_matching_ack_and_pending_snapshot_is_immutable(self):
        state = self.ready()
        data = album()
        queue_browser_save(state, data)
        data["libraries"].clear()
        self.assertFalse(state["saved"])
        self.assertEqual(state["phase"], "saving")
        self.assertIsNone(consume_browser_event(state, event(state, kind="saved", request_id="stale", event_id="wrong")))
        self.assertEqual(state["phase"], "saving")
        written = consume_browser_event(state, event(state, kind="saved", revision="a" * 64, event_id="ack"))
        self.assertEqual(written, album())
        self.assertTrue(state["saved"])
        self.assertEqual(state["phase"], "ready")

    def test_clear_is_saved_and_new_lifecycle_restores_empty(self):
        state = self.ready(album())
        queue_browser_save(state, empty_library())
        result = consume_browser_event(state, event(state, kind="saved", revision="b" * 64, event_id="clear"))
        fresh = new_browser_state()
        self.assertEqual(consume_browser_event(fresh, event(fresh, data=result, revision="b" * 64)), empty_library())

    def test_conflict_and_quota_keep_pending_for_export_and_block_further_mutation(self):
        for kind, code in (("conflict", None), ("error", "quota")):
            state = self.ready()
            queue_browser_save(state, album())
            consume_browser_event(state, event(state, kind=kind, code=code, event_id=kind))
            self.assertEqual(import_library(export_library(state["pending"])), album())
            with self.assertRaises(LibraryDataError):
                queue_browser_save(state, empty_library())
            request_browser_reload(state)
            self.assertEqual(state["phase"], "loading")
            self.assertEqual(consume_browser_event(state, event(state, data=album("other"), event_id="reload")), album("other"))

    def test_external_updates_apply_only_without_pending_write_and_unknown_data_fail_closed(self):
        state = self.ready()
        self.assertEqual(consume_browser_event(state, event(state, kind="external", data=album(), event_id="tab")), album())
        queue_browser_save(state, empty_library())
        self.assertIsNone(consume_browser_event(state, event(state, kind="external", data=album("other"), event_id="busy")))
        self.assertEqual(state["phase"], "saving")
        request_browser_reload(state)
        self.assertIsNone(consume_browser_event(state, event(state, data={"api_key": "no"}, event_id="bad")))
        self.assertEqual(state["phase"], "error")
        self.assertEqual(state["error"], "invalid")

    def test_invalid_candidate_does_not_mutate_ready_state(self):
        state = self.ready(album())
        before = deepcopy(state)
        with self.assertRaises(LibraryDataError):
            queue_browser_save(state, {"libraries": {"too long" * 100: {}}})
        self.assertEqual(state, before)

    def test_watch_id_reconciles_ack_gap_but_rejects_stale_watcher_event(self):
        state = self.ready()
        old_watch = state["request"]["id"]
        queue_browser_save(state, album())
        consume_browser_event(state, event(state, kind="saved", revision="a" * 64, event_id="ack"))
        watch_id = state["request"]["id"]
        self.assertNotEqual(watch_id, old_watch)
        self.assertIsNone(consume_browser_event(state, event(state, kind="external", data=empty_library(),
                                                           request_id=old_watch, event_id="stale")))
        other = album("other-tab")
        self.assertEqual(consume_browser_event(state, event(state, kind="external", data=other,
                                                           revision="b" * 64, event_id="reconcile")), other)
        self.assertEqual(state["request"]["id"], watch_id)


if __name__ == "__main__":
    unittest.main()
