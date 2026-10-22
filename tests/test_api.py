"""Optional legacy API privacy contract, without model/provider calls."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.fixture
def api_app(monkeypatch):
    import api
    monkeypatch.delenv("ADAPHOTORET_LOCAL_MODE", raising=False)
    return api, TestClient(api.app)

def test_catalog_not_entire_data_directory(api_app):
    api, client = api_app
    response = client.get("/api/all-photos")
    assert response.status_code == 200
    assert len(response.json()["photos"]) == len(api.metadata)
    assert client.get("/photos/travel/private.jpg").status_code == 404
    assert client.get("/photos/../README.md").status_code == 404

def test_heic_is_served_as_browser_compatible_jpeg(api_app):
    _, client = api_app
    response = client.get("/photos/travel/travel_01.HEIC")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")

def test_public_no_key_never_creates_host_client(api_app, monkeypatch):
    api, client = api_app
    monkeypatch.setenv("DEEPSEEK_API_KEY", "host-test-key")
    factory = Mock(side_effect=AssertionError("Host credentials forbidden"))
    search = Mock(return_value=(None, None, None, "No result", [], []))
    monkeypatch.setattr(api, "get_deepseek_client", factory)
    monkeypatch.setattr(api, "search_photos", search)
    assert client.post("/api/search", json={"query": "cat"}).status_code == 200
    factory.assert_not_called()
    assert search.call_args.kwargs["include_explanation"] is False

def test_byok_requires_consent_and_closes_client(api_app, monkeypatch):
    api, client = api_app
    fake = SimpleNamespace(close=Mock())
    factory = Mock(return_value=fake)
    search = Mock(return_value=(None, None, None, "No result", [], []))
    monkeypatch.setattr(api, "get_deepseek_client", factory)
    monkeypatch.setattr(api, "search_photos", search)
    assert client.post("/api/search", json={"query": "cat"},
                       headers={"X-DeepSeek-API-Key": "visitor-test-key"}).status_code == 400
    factory.assert_not_called()
    response = client.post("/api/search", json={"query": "cat"}, headers={
        "X-DeepSeek-API-Key": "visitor-test-key", "X-Allow-External-Processing": "true"})
    assert response.status_code == 200
    factory.assert_called_once_with("visitor-test-key", allow_environment=False)
    fake.close.assert_called_once()

def test_errors_do_not_echo_provider_secrets(api_app, monkeypatch):
    api, client = api_app
    monkeypatch.setattr(api, "search_photos", Mock(side_effect=RuntimeError("secret-test-value")))
    response = client.post("/api/search", json={"query": "cat"})
    assert response.status_code == 503
    assert "secret-test-value" not in response.text


def test_multiturn_api_filters_roles_keeps_current_query_and_resets(api_app, monkeypatch):
    api, client = api_app
    search = Mock(return_value=(None, None, None, "No result", [], []))
    monkeypatch.setattr(api, "search_photos", search)
    history = [{"role": "user", "content": "猫"},
               {"role": "assistant", "content": "准确，请告诉我，山崖"},
               {"role": "system", "content": "北京"},
               {"role": "tool", "content": "瀑布"}]
    assert client.post("/api/search", json={"query": "海边", "messages": history}).status_code == 200
    assert search.call_args.args[0] == "猫\n海边"
    assert client.post("/api/search", json={"query": "狗", "messages": []}).status_code == 200
    assert search.call_args.args[0] == "狗"
    assert client.post("/api/search", json={"query": "瀑布"}).status_code == 200
    assert search.call_args.args[0] == "瀑布"


def test_api_history_is_bounded_and_requires_explicit_valid_roles(api_app, monkeypatch):
    api, client = api_app
    search = Mock()
    monkeypatch.setattr(api, "search_photos", search)
    for history in ([{"content": "猫"}], [{"role": "invalid", "content": "猫"}],
                    [{"role": "user", "content": "猫"}] * 101):
        assert client.post("/api/search", json={"query": "海边", "messages": history}).status_code == 422
    search.assert_not_called()

def test_sample_images_have_no_exif_and_match_provenance():
    import hashlib
    from PIL import Image
    expected = {
        "pet_01.jpg": "5020a6f3b234003dd216eb0ecb3bd3c2cb621352",
        "pet_02.jpg": "02fd58714656e5f6574b3d6cdec8bce7eebc58ad",
        "pet_03.jpg": "451acfe7345e54d32f681f42d26f39cdbba389f8",
        "pet_05.jpg": "2a6457ad90e34f4f26c934bc0b9733d333e31bc8",
        "pet_06.jpg": "dacf29d62f7b803a5d0a62434d7fb2d045208526",
    }
    for name, sha in expected.items():
        path = ROOT / "data" / "pet" / name
        data = path.read_bytes()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == sha
        with Image.open(path) as image:
            assert not image.getexif()
