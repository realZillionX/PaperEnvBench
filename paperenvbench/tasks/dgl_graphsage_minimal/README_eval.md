# dgl_graphsage_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/dmlc/dgl` at commit `c6c874bf7ea085beb04ea1487cfd216a0bacd6c1`.

The full DGL repository route is intentionally pinned, but the check-only artifact uses a deterministic fallback rather than requiring a source build or external graph dataset. The fallback records the expected DGL GraphSAGE route: construct a small graph, include self features in neighborhood aggregation, apply a two-layer mean aggregator, and evaluate node-classification predictions on a controlled toy graph.

The hidden evaluator should accept this as `L4_cpu_deterministic_fallback` when `python verify.py --check-only --json` validates the artifact and all semantic checks pass.
