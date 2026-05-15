# pyg_gcn_minimal

This task verifies a pinned PyTorch Geometric route for a minimal GCN node-classification smoke test. The gold route clones `pyg-team/pytorch_geometric` at `a5b69c37a05561ebb92931b3d586d664a7269585`, installs a CPU PyTorch stack, imports `torch_geometric.data.Data` and `torch_geometric.nn.GCNConv`, and runs a deterministic six-node toy graph.

Expected verifier:

```bash
python verify.py --check-only --json
```

The local check-only verifier validates the recorded artifact and route evidence. It does not redownload datasets or require GPU hardware. A valid attempt must identify the pinned repository, the `Data` graph container, the `GCNConv` entrypoint, and the CPU deterministic fallback boundary for environments where optional PyG native extensions are unavailable.
