# bert_masked_lm_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/google-research/bert`. The pinned repository is a TensorFlow 1.x research codebase whose README still specifies `tensorflow >= 1.11.0`; the gold route records the real `run_pretraining.py`, `modeling.py`, `tokenization.py`, vocab, config, and official BERT-Tiny checkpoint path while using a deterministic check-only masked-LM artifact.

The semantic artifact represents the prompt `the capital of france is [MASK] .` with top prediction `paris`. Hidden evaluation should require repository route evidence and tokenizer evidence, not only a modern `transformers` fallback.
