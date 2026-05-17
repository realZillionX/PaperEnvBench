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

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "test_image.png"
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([72, 56, 188, 204], fill=(35, 35, 35))
    draw.ellipse([112, 96, 152, 136], fill=(220, 220, 220))
    image.save(out_dir / "expected_input.png")

    image_np = np.asarray(image)
    if not torch.cuda.is_available():
        raise RuntimeError("SAM L4 smoke requires CUDA")
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    with torch.no_grad():
        predictor.set_image(image_np)
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[128, 130]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

    selected = int(np.argmax(scores))
    mask = masks[selected].astype(np.uint8) * 255
    Image.fromarray(mask).save(out_dir / "expected_artifact.png")
    summary = {
        "task_id": "sam_mask_minimal",
        "entrypoint": "SamPredictor.predict point prompt",
        "model_type": "vit_b",
        "score": float(scores[selected]),
        "all_scores": [float(item) for item in scores.tolist()],
        "selected_mask_index": selected,
        "mask_pixels": int(masks[selected].sum()),
        "logits_shape": list(logits.shape),
        "image": "artifacts/expected_input.png",
        "mask": "artifacts/expected_artifact.png",
        "device": "cuda",
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
        },
    }
    (out_dir / "expected_artifact.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
