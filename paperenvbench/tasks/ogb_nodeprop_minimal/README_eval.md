# ogb_nodeprop_minimal

This task verifies a pinned Open Graph Benchmark node property prediction route for `https://github.com/snap-stanford/ogb`.

The gold route pins commit `61e9784ca76edeaa6e259ba0f836099608ff0586`, installs the OGB package from that checkout, and exercises `from ogb.nodeproppred import Evaluator` with `Evaluator(name="ogbn-arxiv")` on CPU. The full public route may use `PygNodePropPredDataset(name="ogbn-arxiv")`, but the check-only package deliberately avoids downloading OGB datasets.

The embedded artifact records a deterministic toy citation graph, train / validation / test node split, ground-truth node labels, predicted labels, and the OGB accuracy metric. Import success alone is not sufficient; an accepted result must preserve the node property prediction metric route and reproduce the expected accuracy.

Validated local equivalent:

```bash
bash gold_install.sh
python verify.py --check-only --json
```

Validated PaperEnvBench package check:

```bash
python3 tools/paper_repo_env/validate_task_package.py --task ogb_nodeprop_minimal
```
