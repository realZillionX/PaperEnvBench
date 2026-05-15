# diffusers_tiny_pipeline_minimal

This task verifies a pinned, CPU-safe `diffusers` route for the Diffusers paper codebase. The full gold path installs the pinned repository and runs `DiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-pipe")` with a DDIM scheduler on CPU. The local check-only verifier validates the recorded artifact, route evidence, and semantic image-generation metadata without redownloading model files.

Expected verifier:

```bash
python verify.py --check-only --json
```

The task is not satisfied by import-only success. A valid attempt must identify the pinned repository, the `DiffusionPipeline` entrypoint, the tiny public pipeline, and the scheduler route or an equivalent documented fallback artifact.
