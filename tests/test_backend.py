"""Offline regression tests: no downloaded model, credentials, or paid calls."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    return result


def load(name, file_name, stubs):
    spec = importlib.util.spec_from_file_location(name, ROOT / file_name)
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs), patch.object(sys, "path", [str(ROOT), *sys.path]):
        spec.loader.exec_module(result)
    return result


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.constructor = Mock(name="SentenceTransformer")
        self.explain = Mock(return_value="offline explanation")
        posseg = module("jieba.posseg", cut=lambda text: [])
        self.stubs = {
            "faiss": module("faiss", IndexFlatIP=Mock()),
            "jieba": module("jieba", cut=lambda text: list(text), posseg=posseg),
            "jieba.posseg": posseg,
            "pillow_heif": module("pillow_heif", register_heif_opener=lambda: None),
            "sentence_transformers": module("sentence_transformers", SentenceTransformer=self.constructor),
            "llm_explainer": module("llm_explainer", generate_explanation=self.explain),
        }
        self.env = patch.dict(os.environ, {"ADAPHOTORET_LOCAL_MODE": "0"})
        self.env.start()
        self.core = load("_backend_test_core", "AdaphotoRet_run.py", self.stubs)
        self.core.BASE_DIR = self.base
        self.core.get_model.cache_clear()
        self.core._resources = None
        self.core._resource_signature = None
        self.core.metadata.clear()
        self.core.image_paths.clear()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def write_catalog(self, file_name="metadata.demo.json"):
        (self.base / "sample.jpg").write_bytes(b"fixture")
        path = self.base / file_name
        path.write_text(json.dumps({"sample.jpg": {"description": "cat", "category": "动物"}}))
        return path

    def test_import_does_not_load_model_or_gradio(self):
        self.constructor.assert_not_called()
        self.assertNotIn("gr", self.core.__dict__)

    def test_empty_catalog_does_not_load_model(self):
        resources = self.core.get_search_resources()
        self.assertIsNone(resources["model"])
        self.assertIsNone(resources["vector_index"])
        self.assertEqual(len(self.core.search_photos("cat")), 6)
        self.constructor.assert_not_called()
        self.explain.assert_not_called()

    def test_public_mode_does_not_select_private_metadata(self):
        self.write_catalog("metadata_cache.json")
        with patch.dict(os.environ, {"ADAPHOTORET_METADATA_FILE": "metadata_cache.json"}):
            self.assertEqual(self.core.load_metadata(), {})
        with patch.dict(os.environ, {"ADAPHOTORET_LOCAL_MODE": "1"}):
            self.assertIn("sample.jpg", self.core.load_metadata())

    def test_missing_and_outside_root_images_are_filtered(self):
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / "outside.jpg"
            outside.write_bytes(b"fixture")
            path = self.write_catalog()
            data = json.loads(path.read_text())
            data.update({"missing.jpg": {}, str(outside): {}})
            path.write_text(json.dumps(data))
            self.assertEqual(list(self.core.load_metadata()), ["sample.jpg"])

    def test_model_is_pinned_and_cached(self):
        one = self.core.get_model()
        self.assertIs(one, self.core.get_model())
        self.constructor.assert_called_once_with(
            self.core.MODEL_ID, revision=self.core.MODEL_REVISION,
            trust_remote_code=False, device="cpu",
        )

    def test_complete_local_model_uses_no_network(self):
        folder = self.base / "local_model"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"fixture")
        self.core.get_model()
        self.constructor.assert_called_once_with(
            str(folder), local_files_only=True, trust_remote_code=False, device="cpu",
        )

    def test_empty_vector_index_is_valid(self):
        self.assertEqual(self.core.build_vector_index({}), ([], None))
        self.constructor.assert_not_called()

    def test_explanation_receives_explicit_visitor_client(self):
        client = object()
        results = [{"img_path": "sample.jpg", "score": 0.9, "trace": []}]
        self.core.build_reasoning_markdown("cat", {}, results, client=client, allow_environment=False)
        self.explain.assert_called_once_with(
            "cat", results, self.core.metadata, client=client, allow_environment=False,
        )

    def test_evaluation_can_disable_external_explanation(self):
        self.core.build_reasoning_markdown("cat", {}, [], include_explanation=False)
        self.explain.assert_not_called()

    def test_six_value_search_has_no_api_call_when_disabled(self):
        self.core.image_paths[:] = ["sample.jpg"]
        self.core.metadata["sample.jpg"] = {}
        self.core.model = Mock()
        self.core.model.encode.return_value = np.array([[1.0, 0.0]])
        self.core.vector_index = Mock()
        self.core.vector_index.search.return_value = (np.array([[0.8]]), np.array([[0]]))
        self.core.get_search_resources = Mock(return_value={"image_paths": self.core.image_paths})
        self.core.parse_query = Mock(return_value={"expanded_query": "cat", "terms": {}, "decomposition": {}})
        self.core.rerank_score = Mock(return_value=(0.8, []))
        result = self.core.search_photos("cat", include_explanation=False, allow_environment=False)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[:3], ("sample.jpg", None, None))
        self.explain.assert_not_called()

    def test_legacy_blank_rank_browses_without_loading_model(self):
        self.write_catalog()
        self.assertEqual(self.core.rank_all_photos(""), ["sample.jpg"])
        self.constructor.assert_not_called()
        self.explain.assert_not_called()

    def test_ethnicity_does_not_change_rerank_score(self):
        metadata = {"category": "人物", "main_subjects": {"count": 1, "primary_ethnicity": "private unsupported label"}}
        base_score, _ = self.core.rerank_score({}, metadata, 0.8)
        trait_score, trace = self.core.rerank_score({"person_ethnicity": ["private unsupported label"]}, metadata, 0.8)
        self.assertEqual(base_score, trait_score)
        self.assertFalse(any("ethnicity" in item[0] for item in trace))

    def test_api_handles_six_return_values(self):
        backend = module("AdaphotoRet_run", search_photos=Mock(return_value=("a", "b", "c", "report", [], [])))
        api = load("_backend_test_api", "api_server.py", {"AdaphotoRet_run": backend})
        result = api.search(api.QueryRequest(query="cat"))
        self.assertEqual(result["top1"], "a")
        self.assertIn("results", result)


class LabelingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.api_call = Mock(side_effect=AssertionError("No network calls in offline tests"))
        self.label = load("_backend_test_label", "auto_label.py", {
            "dashscope": module("dashscope", MultiModalConversation=types.SimpleNamespace(call=self.api_call)),
            "requests": module("requests", get=self.api_call),
            "geopy": module("geopy"),
            "geopy.geocoders": module("geopy.geocoders", Nominatim=Mock()),
            "pillow_heif": module("pillow_heif", register_heif_opener=lambda: None),
        })

    def tearDown(self):
        self.api_call.assert_not_called()
        self.temp.cleanup()

    def test_uppercase_heic_is_included(self):
        for name in ["upper.HEIC", "lower.heic", "photo.JPG", "skip.txt"]:
            (self.base / name).touch()
        self.assertEqual({p.name for p in self.label.list_image_files(self.base)},
                         {"upper.HEIC", "lower.heic", "photo.JPG"})

    def test_transmitted_bytes_are_jpeg_and_have_no_exif(self):
        path = self.base / "input.png"
        Image.new("RGBA", (3, 2), (255, 0, 0, 100)).save(path)
        raw = base64.b64decode(self.label.encode_image_to_base64(str(path)))
        with Image.open(BytesIO(raw)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.mode, "RGB")
            self.assertFalse(image.getexif())

    def test_labeling_requires_explicit_send_consent(self):
        with self.assertRaises(SystemExit):
            self.label.main([])

    def test_missing_gps_does_not_force_paid_relabel_when_geocoding_disabled(self):
        existing = {"cat.JPG": {"weather": "inside", "main_subjects": {}, "category": "动物"}}
        self.assertFalse(self.label.should_reanalyze("cat.JPG", existing, False, set()))
        self.assertTrue(self.label.should_reanalyze("cat.JPG", existing, False, set(), enable_geocoding=True))

    def test_normalization_drops_inferred_ethnicity(self):
        result = self.label.normalize_metadata_schema({"category": "人物", "main_subjects": {
            "count": 1, "primary_ethnicity": "unsupported inference", "ethnic_ratios": {"private": 1.0},
        }})
        self.assertEqual(result["main_subjects"]["primary_ethnicity"], "")
        self.assertNotIn("ethnic_ratios", result["main_subjects"])

    def test_geocoding_is_opt_in(self):
        self.label.call_bailian_vl = Mock(return_value={"description": "fixture"})
        self.label.normalize_metadata_schema = lambda data: data
        self.label.get_gps_from_image = Mock()
        result = self.label.analyze_image("fixture.jpg")
        self.label.get_gps_from_image.assert_not_called()
        self.assertEqual(result["geo_reasoning"]["source"], "none")


if __name__ == "__main__":
    unittest.main()
