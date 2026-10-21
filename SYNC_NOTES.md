# Sync and preservation notes

Source: the user's confirmed-complete October 21, 2026 Drive upload, compared with the previous public repository at `d5fe2856d9e129f2f650e033b294f301e5080540`.

- Preserve all nine uploaded Python modules, repairing deployment/runtime/privacy issues without substituting placeholder logic.
- Preserve Git-only React/Vite sources and HTTP/conversion entrypoints; remove embedded credentials and machine-specific paths.
- Add explicit dependencies, setup/deployment documentation, the complete user-authorized 66-photo demonstration dataset, and regression tests.
- Do not publish original private chat history, feedback, personal library definitions, private runtime metadata cache, certificates, ngrok configuration, bytecode, or the 470 MB model weight.
- Do not delete or modify original Drive files. Two same-name `data/travel/travel_06.HEIC` uploads were verified byte-identical; no Drive copy is removed. Changes/renames in travel photos are not inferred to be deletion instructions.
- Git history preserves the previous revision, but is not a safe place for secret backups. No extra secret-containing backup branch is created. Removing sensitive paths from a new tree does not remove prior public exposure.
- All 66 canonical uploaded photos and the original 66-entry retrieval metadata are included publicly with explicit user authorization. The supplied metadata cache is preserved as metadata.demo.json; local metadata_cache.json remains ignored. Only a byte-identical duplicate upload is collapsed to one Git path.
