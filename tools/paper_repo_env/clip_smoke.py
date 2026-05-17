from __future__ import annotations

import json
import shutil
from pathlib import Path

import clip
import torch
from PIL import Image


def main() -> None:
    out_dir = Path("artifacts")
    model_dir = Path("models/clip")
    repo_dir = Path("repo")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    image_path = repo_dir / "CLIP.png"
    if not image_path.exists():
        raise FileNotFoundError("Expected upstream README image at repo/CLIP.png")
    image_artifact = out_dir / "expected_artifact.png"
    shutil.copy2(image_path, image_artifact)

    if not torch.cuda.is_available():
        raise RuntimeError("CLIP L4 smoke requires CUDA")
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(model_dir))
    image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    labels = ["a diagram", "a dog", "a cat"]
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0].tolist()

    result = {
        "task_id": "clip_zeroshot_minimal",
        "entrypoint": "README zero-shot image classification example",
        "model_name": "ViT-B/32",
        "labels": labels,
        "logits_per_image": logits_per_image.detach().cpu().numpy()[0].tolist(),
        "probabilities": {label: float(prob) for label, prob in zip(labels, probs)},
        "top_label": labels[int(max(range(len(probs)), key=lambda idx: probs[idx]))],
        "top_probability": float(max(probs)),
        "device": device,
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
        },
    }
    (out_dir / "expected_artifact.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
