"""Privacy regressions; no provider API calls or model downloads required."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import types
import sys
from contextlib import nullcontext
from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError
import unittest
from photo_query import build_photo_query
from photo_report_narrative import ensure_score_explanations
from clarification_policy import clarification_questions, must_return_results, concise_question
from unittest.mock import Mock, patch
from photo_library_store import (
    empty_library, normalize_library, load_local_library, save_local_library, queue_browser_save,
    new_browser_state, consume_browser_event, LibraryDataError,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tested_explainer", ROOT / "llm_explainer.py")
EXPLAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPLAINER)


def ui_functions(*names, **variables):
    """Load selected pure UI helpers without running the Streamlit app."""
    tree = ast.parse((ROOT / "streamlit_ui.py").read_text())
    if "_run_assistant_with_client" in names and "show_search_results" not in variables:
        names = (*names, "show_search_results")
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in definitions:
        node.decorator_list = []
    scope = {"os": os, "Path": Path, "build_photo_query": build_photo_query,
             "clarification_questions": clarification_questions,
             "must_return_results": must_return_results, "concise_question": concise_question,
             "safe_api_error": EXPLAINER.safe_api_error,
             "ensure_score_explanations": ensure_score_explanations, "metadata": {},
             "empty_library": empty_library, "normalize_library": normalize_library, "load_local_library": load_local_library,
             "save_local_library": save_local_library, "queue_browser_save": queue_browser_save,
             "LibraryDataError": LibraryDataError, **variables}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "ui_helpers", "exec"), scope)
    return scope


class CredentialTests(unittest.TestCase):
    def test_public_factory_cannot_fall_back_to_host_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "host-private-test-key"}):
            with self.assertRaises(ValueError):
                EXPLAINER.get_deepseek_client(allow_environment=False)

    def test_factory_passes_only_explicit_key_and_does_not_mutate_environment(self):
        fake_openai = SimpleNamespace(OpenAI=Mock(return_value=object()))
        with patch.dict("sys.modules", {"openai": fake_openai}), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "host-test-key"}):
            EXPLAINER.get_deepseek_client("visitor-test-key", allow_environment=False)
            self.assertEqual(fake_openai.OpenAI.call_args.kwargs["api_key"], "visitor-test-key")
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "host-test-key")

    def test_errors_never_render_provider_exception_contents(self):
        error = RuntimeError("sensitive-test-key and Authorization header")
        self.assertNotIn("sensitive-test-key", EXPLAINER.safe_api_error(error))
        error.status_code = 401
        self.assertIn("密钥", EXPLAINER.safe_api_error(error))

    def test_public_session_ignores_server_environment_key(self):
        st = SimpleNamespace(session_state={})
        scope = ui_functions("session_api_key", "fine_search_ready", st=st, LOCAL_MODE=False)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "host-private-test-key"}):
            self.assertEqual(scope["session_api_key"](), "")
            self.assertFalse(scope["fine_search_ready"]())
            st.session_state["visitor_deepseek_key"] = "visitor-test-key"
            self.assertFalse(scope["fine_search_ready"]())
            st.session_state["deepseek_consent"] = True
            self.assertTrue(scope["fine_search_ready"]())

    def test_clear_credentials_revokes_consent(self):
        st = SimpleNamespace(session_state={"visitor_deepseek_key": "test-key", "deepseek_consent": True})
        scope = ui_functions("clear_visitor_credentials", st=st)
        scope["clear_visitor_credentials"]()
        self.assertNotIn("visitor_deepseek_key", st.session_state)
        self.assertFalse(st.session_state["deepseek_consent"])

    def test_local_mode_can_use_user_environment(self):
        scope = ui_functions("session_api_key", st=SimpleNamespace(session_state={}), LOCAL_MODE=True)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "local-test-key"}):
            self.assertEqual(scope["session_api_key"](), "local-test-key")

    def test_explanation_uses_request_client_without_environment_fallback(self):
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="真实测试解释"))])))))
        with patch.object(EXPLAINER, "get_deepseek_client", side_effect=AssertionError("must not create host client")):
            result = EXPLAINER.generate_explanation("猫", [{"img_path": "cat.jpg", "trace": [], "score": 90}],
                                                    {"cat.jpg": {"category": "宠物"}},
                                                    client=fake_client, allow_environment=False)
        self.assertEqual(result, "真实测试解释")
        self.assertEqual(fake_client.chat.completions.create.call_count, 1)

    def test_explainer_closes_only_clients_it_creates_even_on_provider_error(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="test explanation"))])
        for fails in (False, True):
            with self.subTest(provider_error=fails):
                client = SimpleNamespace(close=Mock(), chat=SimpleNamespace(completions=SimpleNamespace(
                    create=Mock(return_value=response, side_effect=RuntimeError("private fixture detail") if fails else None))))
                with patch.object(EXPLAINER, "get_deepseek_client", return_value=client):
                    result = EXPLAINER.generate_explanation("query", [{"img_path": "cat.jpg", "trace": [], "score": 1}], {},
                                                            allow_environment=False)
                client.close.assert_called_once_with()
                self.assertNotIn("private fixture detail", result)
        external = SimpleNamespace(close=Mock(), chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))
        EXPLAINER.generate_explanation("query", [{"img_path": "cat.jpg", "trace": [], "score": 1}], {}, client=external)
        external.close.assert_not_called()

    def test_assistant_closes_request_client_on_success_preview_error_and_provider_error(self):
        for outcome in ("success", "preview_error", "provider_error"):
            with self.subTest(outcome=outcome):
                chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="【思考】检索依据【行动】回答：测试答复"))])
                create = Mock(return_value=[chunk], side_effect=RuntimeError("private provider detail") if outcome == "provider_error" else None)
                client = SimpleNamespace(close=Mock(), chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                placeholder = SimpleNamespace(markdown=Mock(), error=Mock())
                st = SimpleNamespace(info=Mock(), error=Mock(), chat_message=lambda *_: nullcontext(), empty=lambda: placeholder)
                preview = Mock(return_value=[], side_effect=RuntimeError("preview detail") if outcome == "preview_error" else None)
                scope = ui_functions("run_assistant", "_run_assistant_with_client", st=st, image_paths=["cat.jpg"],
                                     fine_search_ready=lambda: True, session_api_key=lambda: "visitor-fixture-key",
                                     get_deepseek_client=Mock(return_value=client), preview_search_candidates=preview,
                                     get_current_messages=lambda: [], add_message=Mock(), should_direct_search=lambda _: False,
                                     safe_api_error=EXPLAINER.safe_api_error)
                scope["run_assistant"]("query")
                client.close.assert_called_once_with()
                self.assertEqual(create.call_count, 0 if outcome == "preview_error" else 1)

    def test_assistant_cleanup_failure_does_not_expose_provider_details(self):
        client = SimpleNamespace(close=Mock(side_effect=RuntimeError("private cleanup detail")))
        st = SimpleNamespace(info=Mock(), error=Mock())
        scope = ui_functions("run_assistant", st=st, image_paths=["cat.jpg"], fine_search_ready=lambda: True,
                             session_api_key=lambda: "visitor-fixture-key", get_deepseek_client=Mock(return_value=client),
                             _run_assistant_with_client=Mock(side_effect=RuntimeError("private stream detail")))
        scope["run_assistant"]("query")
        client.close.assert_called_once_with()
        self.assertNotIn("private", st.error.call_args.args[0])


class SessionStorageTests(unittest.TestCase):
    def test_public_history_and_libraries_never_read_or_write_files(self):
        state = new_browser_state()
        consume_browser_event(state, {"kind": "loaded", "event_id": "load", "request_id": state["request"]["id"],
                                      "revision": None, "data": empty_library()})
        class Session(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        st = SimpleNamespace(session_state=Session(photo_library_storage=state))
        scope = ui_functions("load_all_conversations", "save_all_conversations", "load_photo_library_data",
                             "save_photo_library_data", LOCAL_MODE=False, st=st)
        with patch("builtins.open", side_effect=AssertionError("public state must not access files")):
            self.assertEqual(scope["load_all_conversations"](), {})
            scope["save_all_conversations"]({"one-session": {}})
            self.assertEqual(scope["load_photo_library_data"](), {"photo_tags": {}, "libraries": {}})
            scope["save_photo_library_data"]({"libraries": {"private": {}}})
            self.assertEqual(state["phase"], "saving")

    def test_distinct_sessions_receive_distinct_empty_libraries(self):
        scope = ui_functions("load_photo_library_data", LOCAL_MODE=False)
        a, b = scope["load_photo_library_data"](), scope["load_photo_library_data"]()
        a["libraries"]["one-user"] = {"photos": []}
        self.assertEqual(b["libraries"], {})

    def test_search_tool_passes_request_client_and_disallows_host_key(self):
        search = Mock(return_value=(None, None, None, "empty", [], []))
        scope = ui_functions("search_photos_tool", search_photos=search)
        client = object()
        scope["search_photos_tool"]("query", client)
        search.assert_called_once_with("query", client=client, allow_environment=False, include_explanation=False)

    def test_search_failure_does_not_fabricate_images_or_expose_error(self):
        scope = ui_functions("search_photos_tool", search_photos=Mock(side_effect=RuntimeError("secret-test-value")))
        result = scope["search_photos_tool"]("query")
        self.assertEqual((result["top1"], result["top2"], result["top3"]), (None, None, None))
        self.assertNotIn("secret-test-value", result["report"])

    def test_no_credential_function_has_global_cache_decorator(self):
        tree = ast.parse((ROOT / "streamlit_ui.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and any(word in node.name for word in ("client", "api_key", "credential")):
                self.assertFalse(node.decorator_list, node.name)


class PhotoRenderingTests(unittest.TestCase):
    def image_helpers(self, root):
        return ui_functions("cached_image_tile", Image=Image, ImageOps=ImageOps, BytesIO=BytesIO,
                            UnidentifiedImageError=UnidentifiedImageError,
                            resolve_photo_path=lambda path: str(root / path))

    def test_unicode_jpeg_reference_preserves_aspect_ratio_for_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "中文 相册" / "猫咪_01.jpg"
            path.parent.mkdir()
            Image.new("RGB", (80, 20), "orange").save(path)
            scope = self.image_helpers(root)
            data = scope["cached_image_tile"]("中文 相册/猫咪_01.jpg", path.stat().st_mtime, crop=False)
            with Image.open(BytesIO(data)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (80, 20))

    @unittest.skipUnless(importlib.util.find_spec("pillow_heif"), "HEIC codec not installed")
    def test_uppercase_heic_unicode_reference_converts_to_browser_jpeg(self):
        from pillow_heif import register_heif_opener
        register_heif_opener()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "旅行照片_01.HEIC"
            Image.new("RGB", (32, 24), "blue").save(path, format="HEIF")
            scope = self.image_helpers(root)
            data = scope["cached_image_tile"](path.name, path.stat().st_mtime)
            self.assertTrue(data.startswith(b"\xff\xd8"))
            with Image.open(BytesIO(data)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (520, 390))

    def test_missing_photo_reference_returns_none_not_a_fake_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope = self.image_helpers(Path(tmp))
            self.assertIsNone(scope["cached_image_tile"]("不存在.HEIC", 0))

    def test_all_ui_image_calls_use_converted_bytes(self):
        tree = ast.parse((ROOT / "streamlit_ui.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "image"]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, "tile")

    @unittest.skipUnless((ROOT / "metadata.demo.json").is_file(), "full demo catalogue not staged")
    def test_demo_metadata_references_exist_and_render(self):
        from pillow_heif import register_heif_opener
        register_heif_opener()
        catalogue = json.loads((ROOT / "metadata.demo.json").read_text())
        self.assertTrue(catalogue)
        scope = self.image_helpers(ROOT)
        for reference in catalogue:
            with self.subTest(reference=reference):
                path = (ROOT / reference).resolve()
                self.assertTrue(path.is_relative_to(ROOT))
                self.assertTrue(path.is_file(), f"Missing catalogue photo: {reference}")
                data = scope["cached_image_tile"](reference, path.stat().st_mtime, crop=False)
                self.assertIsNotNone(data, f"Cannot decode catalogue photo: {reference}")
                with Image.open(BytesIO(data)) as image:
                    self.assertEqual(image.format, "JPEG")


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Streamlit not installed")
class StreamlitInteractionTests(unittest.TestCase):
    """Actual Streamlit UI runtime with a no-network catalogue backend fixture."""
    def setUp(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        self.st, self.AppTest = st, AppTest
        st.cache_resource.clear()
        catalogue_path = ROOT / "metadata.demo.json"
        catalogue = json.loads(catalogue_path.read_text())
        backend = types.ModuleType("AdaphotoRet_run")
        backend.load_metadata = lambda: catalogue
        backend.resolve_photo_path = lambda path: str(ROOT / path)
        backend.parse_query = backend.rerank_score = Mock(side_effect=AssertionError("not needed for catalogue"))
        backend.get_search_resources = Mock(side_effect=RuntimeError("model unavailable in this fixture"))
        backend.search_photos = Mock(side_effect=AssertionError("no paid or semantic search in UI fixture"))
        self.previous_backend = sys.modules.get("AdaphotoRet_run")
        sys.modules["AdaphotoRet_run"] = backend
        self.environment_patch = patch.dict(os.environ, {"ADAPHOTORET_LOCAL_MODE": "0", "DEEPSEEK_API_KEY": "host-fixture-secret"})
        self.environment_patch.start()
        self.backend = backend
        self.backend_patch = patch("backend_resources._load_backend", return_value=backend)
        self.backend_patch.start()
        # Streamlit 1.54 AppTest mocks the entire Runtime, including the v2
        # component registry. Supply its real registry to verify protobuf
        # mounting instead of allowing MagicMock strings to crash the app.
        from streamlit.components.v2.component_manager import BidiComponentManager
        app_test_module = importlib.import_module("streamlit.testing.v1.app_test")
        real_mock, manager = app_test_module.MagicMock, BidiComponentManager()
        def runtime_mock(*args, **kwargs):
            runtime = real_mock(*args, **kwargs)
            runtime.bidi_component_registry = manager
            return runtime
        self.runtime_patch = patch.object(app_test_module, "MagicMock", side_effect=runtime_mock)
        self.runtime_patch.start()
        self.registry_patch = patch("streamlit.components.v2.get_bidi_component_manager", return_value=manager)
        self.registry_patch.start()
        import browser_library
        importlib.reload(browser_library)
        # AppTest does not execute JavaScript. Model the documented component
        # events here; JS protocol/transaction behavior has separate tests.
        def browser_event(request):
            if request["op"] == "load":
                return {"kind": "loaded", "event_id": request["id"], "request_id": request["id"],
                        "revision": None, "data": empty_library()}
            if request["op"] == "save":
                return {"kind": "saved", "event_id": request["id"], "request_id": request["id"], "revision": "a" * 64}
        from browser_library import browser_library_bridge
        self.actual_browser_bridge = browser_library_bridge
        self.browser_patch = patch("browser_library.browser_library_bridge", side_effect=browser_event)
        self.browser_bridge = self.browser_patch.start()

    def tearDown(self):
        self.browser_patch.stop()
        self.registry_patch.stop()
        self.runtime_patch.stop()
        self.backend_patch.stop()
        self.st.cache_resource.clear()
        self.environment_patch.stop()
        if self.previous_backend is None:
            sys.modules.pop("AdaphotoRet_run", None)
        else:
            sys.modules["AdaphotoRet_run"] = self.previous_backend

    def app(self):
        app = self.AppTest.from_file(str(ROOT / "streamlit_ui.py"), default_timeout=180).run()
        self.assertEqual(list(app.exception), [])
        return app

    def test_initial_browse_no_model_no_host_key_and_consent_gates_chat(self):
        app = self.app()
        self.backend.get_search_resources.assert_not_called()
        self.assertEqual(len(app.get("imgs")), min(12, len(self.backend.load_metadata())))
        self.assertEqual(app.text_input(key="visitor_deepseek_key").value, "")
        app.radio[0].set_value("精细搜索").run()
        self.assertTrue(app.chat_input[0].disabled)
        app.text_input(key="visitor_deepseek_key").set_value("visitor-fixture-key").run()
        self.assertTrue(app.chat_input[0].disabled)
        app.checkbox(key="deepseek_consent").check().run()
        self.assertFalse(app.chat_input[0].disabled)
        next(button for button in app.button if button.label == "清除密钥并撤回同意").click().run()
        self.assertEqual(app.text_input(key="visitor_deepseek_key").value, "")
        self.assertFalse(app.checkbox(key="deepseek_consent").value)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual(list(app.exception), [])

    def test_report_import_survives_cached_predeployment_llm_module(self):
        old_module = types.ModuleType("llm_explainer")
        for name in ("get_deepseek_client", "safe_api_error", "generate_explanation"):
            setattr(old_module, name, getattr(EXPLAINER, name))
        self.assertFalse(hasattr(old_module, "ensure_score_explanations"))
        with patch.dict(sys.modules, {"llm_explainer": old_module}):
            app = self.app()
            app.radio[0].set_value("精细搜索").run()
            self.assertEqual(list(app.exception), [])
            self.assertTrue(app.chat_input[0].disabled)

    def test_library_and_credentials_are_isolated_between_two_sessions(self):
        first = self.app()
        first.text_input(key="visitor_deepseek_key").set_value("first-session-fixture-key").run()
        first.radio[0].set_value("照片库").run()
        next(widget for widget in first.text_input if widget.label == "照片库名称").set_value("只属于第一个会话").run()
        next(button for button in first.button if button.label == "保存照片库").click().run()
        self.assertIn("只属于第一个会话", first.session_state["photo_library_data"]["libraries"])
        second = self.app()
        self.assertEqual(second.text_input(key="visitor_deepseek_key").value, "")
        self.assertEqual(second.session_state["photo_library_data"]["libraries"], {})
        self.assertEqual(list(first.exception), [])
        self.assertEqual(list(second.exception), [])

    def test_library_mutations_are_disabled_until_browser_restore(self):
        self.browser_bridge.side_effect = self.actual_browser_bridge
        app = self.app()
        app.radio[0].set_value("照片库").run()
        for label in ("保存照片库", "加入照片库并打标签", "清空自定义照片库和标签", "合并导入备份"):
            self.assertTrue(next(button for button in app.button if button.label == label).disabled)
        self.assertEqual(app.session_state["photo_library_data"], empty_library())
        self.assertTrue(any("正在读取" in item.value for item in app.info))

    def test_preupgrade_session_library_migrates_after_restore_not_before(self):
        app = self.AppTest.from_file(str(ROOT / "streamlit_ui.py"), default_timeout=180)
        old = {"photo_tags": {}, "libraries": {"升级前相册": {"photos": [], "tags": ["保留"]}}}
        app.session_state["photo_library_data"] = old
        app.run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["photo_library_data"], old)
        self.assertTrue(app.session_state["photo_library_storage"]["saved"])
        operations = [call.args[0]["op"] for call in self.browser_bridge.call_args_list]
        self.assertLess(operations.index("load"), operations.index("save"))

    def test_overlimit_preupgrade_session_is_exportable_without_crashing_or_overwriting_browser(self):
        app = self.AppTest.from_file(str(ROOT / "streamlit_ui.py"), default_timeout=180)
        old = {"photo_tags": {}, "libraries": {"x" * 201: {"photos": [], "tags": []}}}
        app.session_state["photo_library_data"] = old
        app.run()
        app.radio[0].set_value("照片库").run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(json.loads(app.session_state["photo_library_legacy_backup"]), old)
        self.assertEqual(app.session_state["photo_library_data"], empty_library())
        self.assertNotIn("save", [call.args[0]["op"] for call in self.browser_bridge.call_args_list])

    def test_library_simulated_refresh_and_confirmed_delete_restore(self):
        from copy import deepcopy
        stored = empty_library()
        def bridge(request):
            nonlocal stored
            if request["op"] == "load":
                return {"kind": "loaded", "event_id": request["id"], "request_id": request["id"],
                        "data": deepcopy(stored), "revision": "c" * 64}
            if request["op"] == "save":
                stored = deepcopy(request["data"])
                return {"kind": "saved", "event_id": request["id"], "request_id": request["id"], "revision": "c" * 64}
        self.browser_bridge.side_effect = bridge
        first = self.app()
        first.radio[0].set_value("照片库").run()
        next(widget for widget in first.text_input if widget.label == "照片库名称").set_value("新建").run()
        next(button for button in first.button if button.label == "保存照片库").click().run()
        self.assertIn("新建", stored["libraries"])
        fresh = self.app()  # New Python session; explicit fixture simulates same browser document.
        fresh.radio[0].set_value("照片库").run()
        next(widget for widget in fresh.selectbox if widget.label == "编辑照片库").set_value("新建").run()
        self.assertTrue(next(button for button in fresh.button if button.label == "删除照片库").disabled)
        fresh.checkbox(key="confirm_delete_library:新建").check().run()
        next(button for button in fresh.button if button.label == "删除照片库").click().run()
        self.assertNotIn("新建", stored["libraries"])
        again = self.app()
        self.assertEqual(again.session_state["photo_library_data"], empty_library())
        self.assertEqual(list(fresh.exception), [])

    def test_library_clear_resets_confirmation_and_keeps_catalogue(self):
        app = self.app()
        app.session_state["photo_library_data"] = {"photo_tags": {"x.jpg": ["tag"]},
                                                    "libraries": {"album": {"photos": ["x.jpg"], "tags": []}}}
        app.radio[0].set_value("照片库").run()
        self.assertTrue(app.button(key="clear_photo_library").disabled)
        app.checkbox(key="confirm_clear_photo_library").check().run()
        app.button(key="clear_photo_library").click().run()
        self.assertEqual(app.session_state["photo_library_data"], empty_library())
        self.assertFalse(app.checkbox(key="confirm_clear_photo_library").value)
        self.assertTrue(len(app.get("imgs")) > 0)
        self.assertEqual(list(app.exception), [])

    def test_invalid_tag_never_replaces_valid_library_or_loses_backup_data(self):
        app = self.app()
        app.radio[0].set_value("照片库").run()
        app.session_state["selected_tag_paths"] = [next(iter(self.backend.load_metadata()))]
        next(widget for widget in app.text_input if widget.label == "本次添加的标签").set_value("x" * 201).run()
        next(button for button in app.button if button.label == "加入照片库并打标签").click().run()
        self.assertEqual(app.session_state["photo_library_data"], empty_library())
        self.assertTrue(any("本次修改未保存" in item.value for item in app.error))
        self.assertEqual(list(app.exception), [])

    def test_new_conversation_and_switching_keep_photo_clues_separate(self):
        app = self.app()
        app.radio[0].set_value("精细搜索").run()
        first_id = app.session_state["current_conv_id"]
        first = app.session_state["conversations"][first_id]
        first["messages"].append({"role": "user", "content": "海边的猫"})
        next(button for button in app.button if "新建对话" in button.label).click().run()
        second_id = app.session_state["current_conv_id"]
        self.assertNotEqual(first_id, second_id)
        second = app.session_state["conversations"][second_id]
        self.assertEqual(build_photo_query(second["messages"]), "")
        second["messages"].append({"role": "user", "content": "山上的瀑布"})
        app.session_state["current_conv_id"] = first_id
        app.run()
        self.assertEqual(build_photo_query(app.session_state["conversations"][first_id]["messages"]), "海边的猫")
        self.assertEqual(build_photo_query(app.session_state["conversations"][second_id]["messages"]), "山上的瀑布")
        fresh_session = self.app()
        self.assertEqual(build_photo_query(fresh_session.session_state["conversations"][fresh_session.session_state["current_conv_id"]]["messages"]), "")
        self.assertEqual(list(app.exception), [])
        self.assertEqual(list(fresh_session.exception), [])

    def test_failed_semantic_download_is_visible_not_a_stacktrace_or_fake_results(self):
        app = self.app()
        next(widget for widget in app.text_input if "输入关键词" in widget.label).set_value("fixture query").run()
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("语义模型暂时不可用" in error.value for error in app.error))
        self.assertEqual(len(app.get("imgs")), 0)

    def test_saved_numeric_report_gains_explanations_on_rerun_without_search_or_key(self):
        app = self.app()
        app.radio[0].set_value("精细搜索").run()
        path = next(iter(self.backend.load_metadata()))
        report = ("### 查询「晋祠大门」的可解释检索结果\n\n#### 推理链（Top 3）\n"
                  f"1. `{path}` | 最终得分: 76\n"
                  "[base_semantic_similarity +0.379] [entity_keyword_match +0.150]\n"
                  "📌 向量语义相似 0.7582 × 0.50 | 命中实体: 大门\n")
        conv = app.session_state["conversations"][app.session_state["current_conv_id"]]
        conv["messages"].append(dict(role="assistant", content="旧候选", images=[path], report=report, kind="results"))
        app.run()
        self.assertEqual(list(app.exception), [])
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("逐图评分说明", rendered)
        self.assertIn("已有识别", rendered)
        self.assertIn("命中实体: 大门", rendered)
        self.assertEqual(conv["messages"][-1]["report"], report)
        self.backend.search_photos.assert_not_called()
        self.backend.get_search_resources.assert_not_called()
        self.assertTrue(app.chat_input[0].disabled)

    def test_three_question_runtime_cap_survives_reruns_and_target_switching(self):
        import numpy as np
        paths = list(self.backend.load_metadata())[:2]
        catalogue = {
            paths[0]: {"scene": "海边", "description": "一只黑色猫戴草帽"},
            paths[1]: {"scene": "屋顶", "description": "两只白色猫没有帽子"},
        }
        self.backend.get_search_resources = lambda: {
            "model": SimpleNamespace(encode=lambda *_, **__: np.zeros((1, 2))),
            "vector_index": SimpleNamespace(search=lambda *_: (np.array([[0.4, 0.3]]), np.array([[0, 1]]))),
            "image_paths": paths, "metadata": catalogue,
        }
        self.backend.parse_query = lambda query: {"terms": {}, "expanded_query": query}
        self.backend.rerank_score = lambda *_: (45, [])
        self.backend.search_photos = Mock(return_value=(*paths, None, "离线候选报告", [], []))
        questions = ["是在海边还是屋顶？", "猫是黑色还是白色？", "照片里是一只还是两只猫？", "是在海边还是屋顶？"]
        chunks = [[SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="【行动】提问：" + q))])]
                  for q in questions]
        create = Mock(side_effect=chunks)
        client = SimpleNamespace(close=Mock(), chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch("llm_explainer.get_deepseek_client", return_value=client):
            app = self.app()
            app.radio[0].set_value("精细搜索").run()
            app.text_input(key="visitor_deepseek_key").set_value("offline-fixture-key").run()
            app.checkbox(key="deepseek_consent").check().run()
            first_id = app.session_state["current_conv_id"]
            for clue in ["猫", "屋顶", "白色", "两只"]:
                app.chat_input[0].set_value(clue).run()
                self.assertEqual(list(app.exception), [])
            first = app.session_state["conversations"][first_id]["messages"]
            self.assertEqual(clarification_questions(first), questions[:3])
            self.assertEqual(first[-1]["kind"], "results")
            self.assertEqual(first[-1]["images"], paths)
            self.assertEqual(create.call_count, 3)
            self.backend.search_photos.assert_called_once_with(
                "猫\n屋顶\n白色\n两只", client=None, allow_environment=False, include_explanation=False)
            next(button for button in app.button if "新建对话" in button.label).click().run()
            second_id = app.session_state["current_conv_id"]
            app.chat_input[0].set_value("猫").run()
            self.assertEqual(len(clarification_questions(app.session_state["conversations"][second_id]["messages"])), 1)
            app.session_state["current_conv_id"] = first_id
            app.run()
            app.chat_input[0].set_value("戴帽子").run()
            self.assertEqual(create.call_count, 4)  # only the new target received another question
            self.assertEqual(len(clarification_questions(app.session_state["conversations"][first_id]["messages"])), 3)
            self.assertEqual(list(app.exception), [])


if __name__ == "__main__":
    unittest.main()
