# fasttext_text_classification_minimal

This task checks whether an agent can reproduce a minimal CPU text classification workflow from the pinned `facebookresearch/fastText` repository.

The gold path uses commit `1142dc4c4ecbc19cc16eee5cdd28472e689267e6`, installs a C++ build toolchain, builds the native `./fasttext` CLI with `make`, and installs the Python bindings from the same checkout with pybind11 and NumPy.

The semantic smoke uses a controlled eight-line training file with `__label__sports` and `__label__tech`, runs `./fasttext supervised`, evaluates two held-out examples with `./fasttext test`, and records `predict-prob` outputs for one sports sentence and one tech sentence. CLI help or import success alone is not sufficient; the accepted artifact must show training, evaluation, prediction, and the Python API route.

Validated local equivalent:

```bash
bash gold_install.sh
python verify.py --check-only --json
```

Validated PaperEnvBench package check:

```bash
python3 tools/paper_repo_env/validate_task_package.py --task fasttext_text_classification_minimal
```
