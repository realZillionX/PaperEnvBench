# latent_diffusion_sample_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/CompVis/latent-diffusion` at commit `a506df5756472e2ebaf9078affdde2c4f1502cd4`.

The gold path records the real CompVis latent diffusion route without requiring hidden evaluators to download the large `models/ldm/text2img-large/model.ckpt` checkpoint. A passing artifact must link the `scripts/txt2img.py` entrypoint, `configs/latent-diffusion/txt2img-1p4B-eval.yaml`, `ldm.models.diffusion.ddim.DDIMSampler`, and `ldm.models.autoencoder.AutoencoderKL`.

## Gold install

```bash
bash gold_install.sh
```

## Verification

```bash
python verify.py --check-only --json
```

The minimum accepted result is `L4_fallback`: the artifact documents the checkpoint boundary and verifies the config / sampler / autoencoder route, but does not claim full text-to-image image synthesis without the public checkpoint.
