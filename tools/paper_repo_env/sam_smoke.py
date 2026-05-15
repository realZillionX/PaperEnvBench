from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry


def main() -> None:
    checkpoint = Path("models/sam/sam_vit_b_01ec64.pth")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    out_dir = Path("outputs/sam_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "test_image.png"
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((68, 68, 188, 188), fill=(30, 144, 255))
    image.save(image_path)

    image_np = np.asarray(image)
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam.to(device="cpu")
    predictor = SamPredictor(sam)

    with torch.no_grad():
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[128, 128]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )

    mask = masks[0].astype(np.uint8) * 255
    Image.fromarray(mask).save(out_dir / "mask.png")
    summary = {
        "score": float(scores[0]),
        "mask_pixels": int(masks[0].sum()),
        "image": str(image_path),
        "mask": str(out_dir / "mask.png"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
