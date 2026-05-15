# fairseq_translation_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/facebookresearch/fairseq` and the paper "fairseq: A Fast, Extensible Toolkit for Sequence Modeling".

The gold route records the real sequence-modeling entrypoints used for translation: `fairseq_cli/generate.py`, `fairseq.tasks.translation.TranslationTask`, and the Transformer model path. Hidden check-only verification validates a deterministic German-to-English toy semantic artifact so evaluators can distinguish a real `fairseq-generate` route from a generic import-only setup. Full agent attempts may materialize the toy parallel text, run `fairseq-preprocess`, and call `fairseq-generate`; the check-only artifact avoids dataset downloads and native-extension rebuilds during package validation.
