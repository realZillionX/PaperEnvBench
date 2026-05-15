# recbole_sequential_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/RUCAIBox/RecBole` and the paper "RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms".

The gold route records the real RecBole sequential recommendation path: `run_recbole.py`, `recbole.quick_start.run_recbole`, `recbole.data.dataset.sequential_dataset.SequentialDataset`, and `recbole.model.sequential_recommender.sasrec.SASRec`. Hidden check-only verification validates a deterministic toy next-item recommendation artifact with `Hit@10`、`MRR@10` and `NDCG@10` so evaluators can distinguish a real sequential recommendation route from a generic import-only setup. Full agent attempts may materialize the toy `.inter` file and run a short SASRec training smoke; the check-only artifact avoids public dataset downloads and long training during package validation.
