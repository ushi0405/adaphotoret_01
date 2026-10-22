"""Content-versioned, read-only backend resources for Streamlit reruns."""
from importlib.util import module_from_spec, spec_from_file_location

import streamlit as st


def _load_backend(source: bytes, source_path: str):
    # Never reload a module that another visitor's in-flight query may be using.
    # Compile the exact version used in the cache key, not a timestamp-based pyc.
    spec = spec_from_file_location("AdaphotoRet_run", source_path)
    backend = module_from_spec(spec)
    exec(compile(source, source_path, "exec"), backend.__dict__)
    return backend


@st.cache_resource(max_entries=1, show_spinner=False)
def load_core_resources(source: bytes, source_path: str):
    # Streamlit hashes the source bytes: changing parser/flattening/scoring code
    # creates a fresh module and lazy index even if file mtime is unchanged.
    # No visitor queries, history, credentials, or consent enter this cache.
    backend = _load_backend(source, source_path)
    catalogue = backend.load_metadata()
    return {"backend": backend, "metadata": catalogue, "image_paths": list(catalogue)}
