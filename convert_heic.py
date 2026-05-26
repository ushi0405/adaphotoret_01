from pillow_heif import register_heif_opener
register_heif_opener()
from PIL import Image
import os

PHOTOS_BASE = r"C:\Users\Lenovo\Desktop\AdaphotoRet_523\data"
for root, _, files in os.walk(PHOTOS_BASE):
    for file in files:
        if file.lower().endswith(('.heic', '.heif')):
            src = os.path.join(root, file)
            dst = os.path.splitext(src)[0] + '.jpg'
            if not os.path.exists(dst):
                img = Image.open(src)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(dst, 'JPEG', quality=85)
                print(f"转换: {src} -> {dst}")