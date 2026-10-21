"""Integrity checks for the complete explicitly authorized demonstration corpus."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_complete_corpus_matches_manifest_and_metadata():
    manifest = json.loads((ROOT / "DATA_MANIFEST.json").read_text())
    metadata_bytes = (ROOT / "metadata.demo.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    assert manifest["photo_count"] == 66
    assert len(manifest["files"]) == len(metadata) == 66
    assert manifest["photo_bytes"] == 102321341
    assert hashlib.sha256(metadata_bytes).hexdigest() == manifest["metadata_sha256"]
    files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "data").rglob("*") if p.is_file()}
    assert files == set(metadata) == {entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        content = (ROOT / entry["path"]).read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        assert hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() == entry["git_blob_sha1"]

def test_all_original_images_decode_including_uppercase_heic():
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    manifest = json.loads((ROOT / "DATA_MANIFEST.json").read_text())
    heic_count = 0
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        heic_count += path.suffix == ".HEIC"
        with Image.open(path) as image:
            image.load()
            assert image.width > 0 and image.height > 0
    assert heic_count == 40
