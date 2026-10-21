# Retrieval model

- Upstream: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Pinned revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
- Upstream model license: **Apache-2.0**, as declared in its model card.
- Output: 384-dimensional sentence embeddings; CPU is the default.
- Loading uses `trust_remote_code=False`. The project does not force a third-party mirror.

`python download_model.py` downloads and saves the model into ignored `local_model/`. The full model weights are not committed: the uploaded weight file is about 470 MB, above GitHub's ordinary 100 MiB per-file limit. The original Drive copy is retained separately. If `local_model/` is complete it is used offline; otherwise the fixed upstream revision is downloaded on first semantic query.

Keep the upstream model card/license notices with redistributed model assets. The model license does not grant a license to the project's original photos or code. Runtime RAM/disk and outbound access must be available on the hosting platform; startup browsing does not require model weights.
