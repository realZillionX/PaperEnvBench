# guided_diffusion_sample_minimal

This task verifies a minimal, pinned reproduction route for OpenAI `guided-diffusion` at commit `22e0df8183507e13a7813f8d38d51b072ca1e67c`.

The hidden check is CPU-safe. It records the real classifier-guided sampling route from the upstream README and validates a deterministic fallback semantic artifact instead of requiring multi-GB ImageNet checkpoints during package validation. A full L4 run may replace the fallback image with samples produced by `scripts/classifier_sample.py` using the pinned `64x64_diffusion.pt` and `64x64_classifier.pt` checkpoints.
