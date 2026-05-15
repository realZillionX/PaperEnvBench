from __future__ import annotations

import json
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image, ImageDraw


def main() -> None:
    out_dir = Path("outputs/clip_smoke")
    model_dir = Path("models/clip")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "test_image.png"
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((54, 54, 170, 170), outline="black", width=8)
    draw.text((74, 96), "CLIP", fill="black")
    image.save(image_path)

    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(model_dir))
    image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    labels = ["a white image with a black square", "a photo of a dog", "a handwritten equation"]
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    result = {label: float(prob) for label, prob in zip(labels, probs)}
    (out_dir / "label_probs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
