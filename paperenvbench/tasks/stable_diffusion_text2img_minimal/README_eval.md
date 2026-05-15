# stable_diffusion_text2img_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/CompVis/stable-diffusion`. The gold path pins the real repository, records the `scripts/txt2img.py` entrypoint, `configs/stable-diffusion/v1-inference.yaml` model config, checkpoint location, and CreativeML Open RAIL-M license gate.

The check-only artifact does not download `sd-v1-4.ckpt` or run full 512x512 inference. Stable Diffusion v1 weights require explicit license acceptance and the reference script expects GPU-oriented model loading. Hidden verification therefore checks the routed txt2img/config/checkpoint/license evidence and a deterministic CPU fallback thumbnail summary.
