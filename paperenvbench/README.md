# PaperEnvBench Package

This package contains the benchmark taxonomy、registries、task packages and evaluator implementation. See the repository-level `README.md` for the runnable workflow.

Key files:

- `taxonomy.yaml`：Paper Environment Taxonomy。
- `evaluator.py`：hidden-evaluator runner for agent attempt directories。
- `registries/task_registry.yaml`：50 task definitions and split / taxonomy metadata。
- `registries/asset_registry.yaml`：gold artifact and checkpoint records。
- `registries/trajectory_registry.yaml`：gold and evaluated trajectory records。
- `tasks/<task_id>/verify.py`：task-local verifier，uniformly supporting `--check-only --json`。
