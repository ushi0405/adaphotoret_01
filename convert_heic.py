"""Optional local HEIC conversion. Originals are never overwritten or deleted."""
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

def convert(directory: Path):
    register_heif_opener()
    count = 0
    for source in sorted(directory.rglob("*")):
        if source.suffix.lower() not in {".heic", ".heif"}:
            continue
        target = source.with_suffix(".jpg")
        if target.exists():
            continue
        with Image.open(source) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(target, "JPEG", quality=85)
        count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent / "data")
    print(f"Converted {convert(parser.parse_args().directory)} files; originals retained.")
