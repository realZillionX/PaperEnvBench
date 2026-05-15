# mmsegmentation_segformer_minimal

This task validates a minimal CPU semantic segmentation route through the pinned OpenMMLab MMSegmentation repository. The gold run fixes `mmsegmentation` at `b040e147adfa027bbc071b624bedf0ae84dfc922`, installs CPU Torch plus `mmengine==0.10.7` and `mmcv-lite==2.1.0`, then executes the official SegFormer B0 config path with a deterministic synthetic RGB image.

The success level is `L4_fallback`. On the `paper-repro-ci-prep` Python 3.12 CPU notebook, `mmcv==2.1.0` has no installable binary wheel and `mmcv-lite` lacks `mmcv._ext`. MMSegmentation imports several unused `mmcv.ops` symbols while registering all models, so the gold verifier injects a narrow unused `mmcv.ops` stub. The stub is not called during the SegFormer forward pass; the executed model path is still the repository's `MixVisionTransformer` backbone and `SegformerHead` built from the official config.

Gold run directory:

```bash
paper-repro:runs/paperenvbench/detection/mmsegmentation_segformer_minimal
```

Remote commands used:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/mmsegmentation_segformer_minimal "bash gold_install.sh"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/mmsegmentation_segformer_minimal ".venv/bin/python <gold verifier payload>"
```

Expected artifacts:

- `artifacts/expected_artifact.json`: structured summary of package versions, fallback reason, model classes, logits statistics, mask histogram, and semantic checks.
- `artifacts/expected_input.png`: deterministic `64x64` RGB input.
- `artifacts/expected_mask.png`: deterministic `64x64` colorized segmentation mask.

Run the local verifier with:

```bash
python verify.py .
```

The verifier checks the artifact JSON and PNG artifacts, not only imports. It requires the pinned commit, SegFormer model class evidence, finite non-degenerate logits, normalized probabilities, a nonempty mask artifact, and the recorded `mmcv` fallback boundary.
