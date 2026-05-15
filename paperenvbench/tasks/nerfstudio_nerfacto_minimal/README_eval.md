# Nerfstudio Nerfacto Minimal

Pinned upstream commit: `50e0e3c70c775e89333256213363badbf074f29d`.

This task is accepted as `L4_cpu_deterministic_fallback`. The artifact records the real Nerfstudio route, including `ns-train nerfacto`, `NerfstudioDataParserConfig`, `VanillaPipelineConfig`, and `NerfactoModelConfig`. The check-only verifier then validates a deterministic tiny NeRF-style ray-marching artifact on CPU.

Run validation:

```bash
python verify.py --check-only --json
```

The gold install script clones the pinned repo and writes dependency / route evidence, but it intentionally does not require full GPU-oriented Nerfstudio training.
