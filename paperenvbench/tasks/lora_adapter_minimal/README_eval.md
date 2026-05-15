# lora_adapter_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/microsoft/LoRA`. The gold path installs the pinned `loralib` package from source, checks the real LoRA API route, and validates a deterministic rank-2 adapter artifact.

The check-only artifact does not require a pretrained transformer checkpoint or a full finetuning run. It records the core semantics expected from the paper repository: `loralib.Linear` freezes the base weight, exposes trainable `lora_A` and `lora_B` parameters, `loralib.mark_only_lora_as_trainable(model)` filters trainable parameters to adapter tensors, and `loralib.lora_state_dict(model)` saves adapter-only checkpoint keys.
