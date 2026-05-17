#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "sam_mask_minimal"
REPO_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
CHECKPOINT_SHA = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
INPUT_IMAGE_SHA = "da65f94ef9ccf941f2a94e53aa9f98c841733952b9183c78212cfb6dc80528fa"
MASK_SHA = "948e23a4f15ff9c03e4d1cd7bda3b3a7705941ac2ffec18d9f852b16ad3a8241"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"summary JSON is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError("summary JSON must be an object")
    return payload


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"{label} mismatch: expected {expected!r}, got {observed!r}")


def require_cuda(payload: dict[str, Any]) -> None:
    require_equal(payload.get("device"), "cuda", "device")
    torch_info = payload.get("torch")
    if not isinstance(torch_info, dict):
        raise AssertionError("torch evidence must be an object")
    if torch_info.get("cuda_available") is not True:
        raise AssertionError("torch.cuda_available must be true")
    torch_cuda = torch_info.get("torch_cuda")
    if not isinstance(torch_cuda, str) or not torch_cuda.startswith("12."):
        raise AssertionError(f"torch_cuda must record CUDA 12.x, got {torch_cuda!r}")
    if "4090" not in str(torch_info.get("gpu_name") or ""):
        raise AssertionError(f"gpu_name must identify the 4090 runtime, got {torch_info.get('gpu_name')!r}")


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def png_header(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"not a PNG file: {path}")
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", chunk_data)
            return {
                "width": width,
                "height": height,
                "bit_depth": bit_depth,
                "color_type": color_type,
                "compression": compression,
                "filter_method": filter_method,
                "interlace": interlace,
            }
    raise AssertionError(f"PNG missing IHDR chunk: {path}")


def read_grayscale_png(path: Path) -> tuple[int, int, list[int]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"not a PNG file: {path}")
    offset = 8
    width = height = bit_depth = color_type = None
    idat_parts: list[bytes] = []
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", chunk_data)
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise AssertionError("mask PNG must use standard compression/filter and no interlace")
            if bit_depth != 8 or color_type != 0:
                raise AssertionError(f"mask PNG must be 8-bit grayscale, got bit_depth={bit_depth}, color_type={color_type}")
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise AssertionError(f"PNG missing IHDR chunk: {path}")
    raw = zlib.decompress(b"".join(idat_parts))
    row_size = width
    expected_len = height * (row_size + 1)
    if len(raw) != expected_len:
        raise AssertionError(f"unexpected mask PNG payload length: {len(raw)} vs {expected_len}")
    rows: list[bytearray] = []
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scan = bytearray(raw[cursor : cursor + row_size])
        cursor += row_size
        prev = rows[-1] if rows else bytearray(row_size)
        recon = bytearray(row_size)
        for idx, value in enumerate(scan):
            left = recon[idx - 1] if idx else 0
            up = prev[idx]
            upper_left = prev[idx - 1] if idx else 0
            if filter_type == 0:
                recon[idx] = value
            elif filter_type == 1:
                recon[idx] = (value + left) & 0xFF
            elif filter_type == 2:
                recon[idx] = (value + up) & 0xFF
            elif filter_type == 3:
                recon[idx] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                recon[idx] = (value + paeth(left, up, upper_left)) & 0xFF
            else:
                raise AssertionError(f"unsupported PNG filter type: {filter_type}")
        rows.append(recon)
    pixels = [value for row in rows for value in row]
    return width, height, pixels


def verify(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    summary_path = artifact_dir / "expected_artifact.json"
    mask_path = artifact_dir / "expected_artifact.png"
    input_path = artifact_dir / "expected_input.png"

    if not summary_path.exists():
        raise AssertionError(f"missing summary JSON: {summary_path}")
    if not mask_path.exists() or mask_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty mask artifact: {mask_path}")
    if not input_path.exists() or input_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty input artifact: {input_path}")

    payload = load_json(summary_path)
    require_equal(payload.get("task_id"), TASK_ID, "task_id")
    require_equal(payload.get("repo_commit"), REPO_COMMIT, "repo_commit")
    require_equal(payload.get("entrypoint"), "SamPredictor.predict point prompt", "entrypoint")
    require_equal(payload.get("model_type"), "vit_b", "model_type")
    require_equal(payload.get("checkpoint_sha256"), CHECKPOINT_SHA, "checkpoint_sha256")
    require_equal(payload.get("input_image_sha256"), INPUT_IMAGE_SHA, "input_image_sha256")
    require_equal(payload.get("mask_sha256"), MASK_SHA, "mask_sha256")
    require_equal(sha256(input_path), INPUT_IMAGE_SHA, "input image sha256")
    require_equal(sha256(mask_path), MASK_SHA, "mask png sha256")
    require_cuda(payload)

    checkpoint_size = payload.get("checkpoint_size_bytes")
    if not isinstance(checkpoint_size, int) or checkpoint_size < 370_000_000:
        raise AssertionError("checkpoint_size_bytes must prove the SAM vit_b checkpoint was cached")

    score = payload.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise AssertionError("summary.score must be finite")
    if not 0.0 <= float(score) <= 1.0:
        raise AssertionError(f"summary.score must be in [0, 1], got {score}")

    all_scores = payload.get("all_scores")
    if not isinstance(all_scores, list) or len(all_scores) < 1:
        raise AssertionError("all_scores must record predictor multimask scores")
    selected = payload.get("selected_mask_index")
    if not isinstance(selected, int) or selected < 0 or selected >= len(all_scores):
        raise AssertionError("selected_mask_index must index all_scores")

    input_header = png_header(input_path)
    if input_header["width"] != 256 or input_header["height"] != 256 or input_header["bit_depth"] != 8 or input_header["color_type"] != 2:
        raise AssertionError(f"input PNG must be 256 x 256 RGB, got {input_header}")

    width, height, pixels = read_grayscale_png(mask_path)
    require_equal([height, width], payload.get("mask_shape"), "mask_shape")
    unique_values = sorted(set(pixels))
    if unique_values != [0, 255]:
        raise AssertionError(f"mask PNG must contain binary 0/255 pixels, got {unique_values}")
    mask_pixels = sum(1 for value in pixels if value > 0)
    require_equal(mask_pixels, payload.get("mask_pixels"), "mask_pixels")
    if not 1000 <= mask_pixels <= 60000:
        raise AssertionError(f"mask_pixels outside expected range: {mask_pixels}")
    if mask_pixels in {0, width * height}:
        raise AssertionError("mask must be nonempty and non-full")

    prompt = payload.get("prompt")
    if not isinstance(prompt, dict) or prompt.get("point_coords") != [[128, 130]] or prompt.get("point_labels") != [1]:
        raise AssertionError("prompt must record the point prompt used by SamPredictor")
    if prompt.get("multimask_output") is not True:
        raise AssertionError("prompt.multimask_output must be true")
    if payload.get("logits_shape") != [3, 256, 256]:
        raise AssertionError(f"logits_shape must prove multimask predictor output, got {payload.get('logits_shape')!r}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": "L4",
        "artifact_dir": str(artifact_dir),
        "checks": {
            "sam_vit_b_checkpoint_sha256_matches": True,
            "sam_predictor_forward_evidence": True,
            "cuda_device_evidence_present": True,
            "input_png_sha256_matches": True,
            "mask_png_sha256_matches": True,
            "mask_pixels_recomputed_from_png": True,
            "mask_area_in_expected_range": True,
            "summary_json_exists": True,
        },
        "observed": {
            "score": float(score),
            "mask_pixels": mask_pixels,
            "mask_shape": [height, width],
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "input_image_sha256": payload.get("input_image_sha256"),
            "mask_sha256": payload.get("mask_sha256"),
            "device": payload.get("device"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(TASK_ROOT / "artifacts"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = verify(Path(args.artifact_dir))
    except AssertionError as exc:
        failure = {"task_id": TASK_ID, "status": "fail", "error": str(exc)}
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
