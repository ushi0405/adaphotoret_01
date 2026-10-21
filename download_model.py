"""Download the pinned Apache-2.0 multilingual retrieval model (no API key)."""
import argparse
from pathlib import Path

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
BASE_DIR = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="local_model")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = BASE_DIR / output
    from sentence_transformers import SentenceTransformer
    print(f"Downloading {MODEL_ID} at pinned revision {MODEL_REVISION}...")
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION,
                                trust_remote_code=False, device="cpu")
    model.save_pretrained(str(output))
    print(f"Model saved to {output}")


if __name__ == "__main__":
    main()
