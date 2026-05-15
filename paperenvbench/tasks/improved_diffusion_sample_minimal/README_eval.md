# improved_diffusion_sample_minimal

This task verifies a CPU-only minimal sampling route for OpenAI `improved-diffusion` at commit `1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5`.

The upstream README sampling command uses `scripts/image_sample.py`, loads a `.pt` checkpoint, constructs a UNet through `improved_diffusion.script_util.create_model_and_diffusion`, and samples through `GaussianDiffusion.p_sample_loop`. Public checkpoints are large and GPU-oriented, so the gold task uses an L4 CPU fallback: create a tiny `32x32` UNet, write and reload its deterministic initialized state_dict, run `p_sample_loop` for 4 diffusion steps, and record the semantic sample artifact.

Required hidden check:

```bash
python verify.py --check-only --json
```

Required artifact:

- `artifacts/expected_artifact.json`: records the pinned repo, model kwargs, diffusion route, checkpoint-loading status, sample shape, uint8 range, sample statistics, and fallback boundary.

Import-only success is insufficient. A valid attempt must preserve the pinned repo route and produce a non-degenerate sample artifact or match the recorded expected artifact in check-only mode.
