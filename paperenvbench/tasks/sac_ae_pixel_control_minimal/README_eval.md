# SAC-AE Pixel Control Minimal

This task targets `denisyarats/pytorch_sac_ae` at commit `7fa560e21c026c04bb8dcd72959ecf4e3424476c` for the paper "Improving Sample Efficiency in Model-Free Reinforcement Learning from Images".

The original route is a pixel-observation SAC-AE training entry point over DeepMind Control via `train.py`, `dmc2gym`, MuJoCo rendering dependencies and a Python 3.6-era CUDA conda stack. The PaperEnvBench gold task therefore uses an `L4_cpu_deterministic_fallback`: it preserves the pinned repository route and validates the SAC-AE pixel encoder, actor, twin critic and pixel decoder evidence without requiring a real MuJoCo render backend or long replay-buffer training.

Gold generation command:

```bash
git clone https://github.com/denisyarats/pytorch_sac_ae repo
git -C repo checkout 7fa560e21c026c04bb8dcd72959ecf4e3424476c
python verify.py --repo-dir repo --output-dir artifacts --json > logs/gold_verify.log 2>&1
```

Check-only command:

```bash
python verify.py --check-only --json
```

Required artifact:

- `artifacts/expected_artifact.json`: pinned repo metadata, environment boundary, SAC-AE route evidence, deterministic pixel-batch semantics and all verifier checks.
