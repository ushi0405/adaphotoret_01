# Demonstration corpus / 完整演示照片集

The user explicitly authorized publishing **all photos uploaded to the AdaphotoRet Drive folder** as the demonstration corpus on October 21, 2026. This snapshot includes all **66 distinct paths**:

- `data/life/`: 12 JPEGs
- `data/pet/`: 14 JPEGs
- `data/travel/`: 40 HEICs
- Total original photo bytes: **102,321,341**

Drive contains 67 photo-file entries because `data/travel/travel_06.HEIC` was uploaded twice under two Drive IDs. Their SHA-256 values match exactly; the Git path contains one canonical original. Neither Drive copy is deleted. No other photo is omitted, resized, converted, or stripped in the repository. Display thumbnails are derived in memory only.

`metadata.demo.json` is a byte-for-byte copy of the supplied `metadata_cache.json`, renamed to distinguish the intentional public retrieval dataset from a mutable/private runtime cache. Its 66 path keys match the 66 original images exactly. Original time/location/OCR/descriptions and other retrieval fields are preserved; no paid API tagging was performed. The public repository and app therefore disclose the authorized photos and their supplied retrieval metadata, potentially including EXIF and location details. No additional image license is asserted.

`DATA_MANIFEST.json` records every photo path, size, SHA-256 and Git blob SHA-1, plus the metadata checksum. It contains no Drive access tokens or private Drive identifiers. These checksums establish the complete uploaded snapshot; this corpus is not labeled a benchmark and no retrieval-accuracy claim is made.

API keys, chat histories, feedback logs, private library state, certificates and runtime bytecode remain excluded. Original files in Drive are unchanged.
