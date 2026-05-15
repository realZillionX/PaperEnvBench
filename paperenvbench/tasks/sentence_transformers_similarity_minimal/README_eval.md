# sentence_transformers_similarity_minimal

This task verifies a pinned Sentence-BERT style sentence-embedding route for `https://github.com/UKPLab/sentence-transformers`.

The gold route is `SentenceTransformer(model_id).encode(sentences)` followed by `sentence_transformers.util.cos_sim`. The full install script records the real clone, checkout, editable install, public checkpoint, and CPU inference path. Check-only verification uses a deterministic lexical embedding artifact so hidden evaluators can validate the semantic contract without downloading model weights.

The expected semantic result is that the two musical-performance sentences have higher cosine similarity than the unrelated finance sentence.
