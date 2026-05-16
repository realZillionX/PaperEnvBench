# Diffusion and Vision-Language Dependency Notes

This note covers the PaperEnvBench diffusion / vision-language / large-checkpoint slice：

- `stable_diffusion_text2img_minimal`
- `latent_diffusion_sample_minimal`
- `guided_diffusion_sample_minimal`
- `improved_diffusion_sample_minimal`
- `diffusers_tiny_pipeline_minimal`
- `blip_caption_minimal`
- `lavis_blip2_minimal`
- `llava_single_image_minimal`
- `imagebind_embedding_minimal`

The probe records whether the runtime can satisfy the dependency surface behind these tasks：Python packages，CUDA-visible tensor paths，checkpoint / license gates，model cache state，and large-model memory boundaries。

## Probe

Run the lightweight probe from the repository root：

```bash
python3 tools/paper_repo_env/probes/diffusion_vlm_probe.py --json
```

Write a reusable evidence file：

```bash
python3 tools/paper_repo_env/probes/diffusion_vlm_probe.py --json --output runs/diffusion_vlm_probe.json
```

Probe one task and fail only on hard dependency blockers：

```bash
python3 tools/paper_repo_env/probes/diffusion_vlm_probe.py --task diffusers_tiny_pipeline_minimal --json --strict
```

The probe does not download checkpoints and does not print token values。It reports Python package imports，CUDA / free VRAM，Hugging Face token/cache visibility，optional `--model-root` summaries，and per-task structured blockers with `code`、`severity`、`message`、`remediation`、and `evidence`。

## Task Matrix

| Task | Runtime / checkpoint dependency | Recommended probe | Success evidence | Common blockers |
| --- | --- | --- | --- | --- |
| `stable_diffusion_text2img_minimal` | Full Stable Diffusion v1 text-to-image requires accepted CreativeML Open RAIL-M weights and a CUDA-capable route。Recommended full-route boundary is at least about \(8\) GB VRAM。 | `diffusion_vlm_probe.py --task stable_diffusion_text2img_minimal --json`，then run the task verifier。Do not use the probe to download gated weights。 | Artifact records `scripts/txt2img.py`、`configs/stable-diffusion/v1-inference.yaml`、checkpoint route、license gate、prompt semantics，and thumbnail metadata。 | Missing `omegaconf` / `einops` / `diffusers`；no CUDA wheel；missing HF token after license acceptance；wrong checkpoint path；VRAM too small for full inference。 |
| `latent_diffusion_sample_minimal` | Full route uses CompVis latent-diffusion text2img-large checkpoint at `models/ldm/text2img-large/model.ckpt`。Recommended full-route boundary is about \(8\) GB VRAM。 | `diffusion_vlm_probe.py --task latent_diffusion_sample_minimal --json`，plus local verifier for route evidence。 | Artifact links `scripts/txt2img.py`、`txt2img-1p4B-eval.yaml`、`LatentDiffusion`、`AutoencoderKL`、DDIM / PLMS route，and checkpoint boundary。 | Missing `omegaconf`、`pytorch_lightning` drift、large checkpoint absent、CUDA unavailable、entrypoint imports assuming old package layout。 |
| `guided_diffusion_sample_minimal` | Full OpenAI classifier-guided route uses public but large `64x64_diffusion.pt` and `64x64_classifier.pt` checkpoints。Recommended full-route boundary is about \(8\) GB VRAM。 | `diffusion_vlm_probe.py --task guided_diffusion_sample_minimal --json`，then verify `blobfile` and torch import before `scripts/classifier_sample.py`。 | Artifact records `classifier_sample.py`、model flags、sample flags、diffusion checkpoint URL、classifier checkpoint URL，and sample-grid metadata。 | Missing `blobfile`；checkpoint download failure；non-CUDA PyTorch；OOM at classifier guidance batch size；using unguided `image_sample.py` as if it satisfied classifier-guided evidence。 |
| `improved_diffusion_sample_minimal` | Gold route uses a tiny deterministic checkpoint；public full-size checkpoints are still a large-checkpoint pressure point。Recommended accelerator route boundary is about \(4\) GB VRAM for small smoke runs。 | `diffusion_vlm_probe.py --task improved_diffusion_sample_minimal --json`，then run the task verifier against tiny sampling artifact。 | Artifact records `create_model_and_diffusion`、`UNetModel`、`SpacedDiffusion`、tiny checkpoint load、sample shape \(1 \times 3 \times 32 \times 32\)，and `.npz` sample output。 | Local editable package path missing；`blobfile` absent；NumPy / torch version drift；claiming full public checkpoint inference without checkpoint evidence。 |
| `diffusers_tiny_pipeline_minimal` | Uses public `hf-internal-testing/tiny-stable-diffusion-pipe`；small checkpoint exercises `diffusers`、`safetensors`、`accelerate`、and HF cache without requiring a large gated model。 | `diffusion_vlm_probe.py --task diffusers_tiny_pipeline_minimal --json --strict`，then tiny pipeline or verifier smoke。 | Artifact records `DiffusionPipeline.from_pretrained`、`StableDiffusionPipeline`、`DDIMScheduler`、pipeline id、\(64 \times 64\) output size，and scheduler timesteps。 | Missing `diffusers` / `safetensors` / `accelerate`；HF cache offline miss；transformers version mismatch；confusing tiny public pipeline with full Stable Diffusion weights。 |
| `blip_caption_minimal` | Full BLIP captioning depends on large caption checkpoint such as `model_base_caption_capfilt_large.pth`。Recommended full-route boundary is about \(6\) GB VRAM。 | `diffusion_vlm_probe.py --task blip_caption_minimal --json`，then verify checkpoint route or artifact。 | Artifact records `models.blip.blip_decoder`、`configs/caption_coco.yaml`、`predict.py`、checkpoint name，and caption semantics for the synthetic image。 | Missing `timm==0.4.12` or `fairscale`；checkpoint URL or local cache absent；torchvision transform mismatch；non-CUDA run misreported as full checkpoint inference。 |
| `lavis_blip2_minimal` | Full BLIP-2 loads a vision encoder plus large language model，for example FLAN-T5 variants。Recommended full-route boundary is at least about \(16\) GB VRAM。 | `diffusion_vlm_probe.py --task lavis_blip2_minimal --json`，then inspect LAVIS config and model-cache route before loading。 | Artifact records `load_model_and_preprocess`、`Blip2T5`、`blip2_pretrain_flant5xl.yaml`、image processor，and deterministic caption semantics。 | Missing `omegaconf` / `timm`；transformers incompatibility；FLAN-T5 checkpoint not cached；VRAM too small；license/source route not recorded。 |
| `llava_single_image_minimal` | Full LLaVA inference needs a base LLM plus vision tower / adapter，often through HF cache and sometimes gated or license-constrained assets。Recommended full-route boundary is at least about \(16\) GB VRAM。 | `diffusion_vlm_probe.py --task llava_single_image_minimal --json`，with HF token visible only as a boolean，then run verifier or full route separately。 | Artifact records conversation template、image token、`llava/model/builder.py` route，processed image size，and answer semantics。 | Missing `sentencepiece`；HF token/license not accepted；base LLM unavailable；CUDA OOM；only preprocessing the image without recording LLaVA conversation/model route。 |
| `imagebind_embedding_minimal` | Full ImageBind uses `imagebind_huge.pth` under a non-commercial checkpoint/license boundary and multimodal torch libraries。Recommended full-route boundary is about \(8\) GB VRAM。 | `diffusion_vlm_probe.py --task imagebind_embedding_minimal --json`，plus module import checks for `torchaudio` and `pytorchvideo`。 | Artifact records `ImageBindModel`、modalities `vision,text,audio`、checkpoint name，and synthetic aligned embedding evidence。 | Missing `torchaudio` / `pytorchvideo`；audio backend issues；checkpoint license not recorded；large checkpoint absent；claiming pretrained embedding quality without checkpoint evidence。 |

## Blocker Interpretation

`missing_required_python_modules` is a hard environment blocker for the selected task。Install the locked requirements or equivalent compatible wheels before rerunning。

`missing_optional_acceleration_modules` is usually a warning。For this slice，`xformers` is an accelerator boundary，while `safetensors` and `accelerate` become practically required for many modern HF routes。

`cuda_not_available` means the run has not proven the CUDA-visible inference route。

`insufficient_free_vram` means the detected free GPU memory is below the task note’s recommended boundary。Lower batch size may still work for some smoke runs，but the artifact should record the memory constraint。

`hf_auth_or_license_gate_unconfirmed` means no `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` was visible。Do not paste token values into logs；only record that license acceptance and runtime token injection were required。

`large_checkpoint_boundary` is informational。It reminds the evaluator author or worker to record checkpoint source、license、path、checksum when available，and whether the actual weights were loaded。
