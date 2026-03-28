#!/usr/bin/env python3
"""
Qwen2-VL-7B offline prescription reader.

Requirements:
  pip install --upgrade torch transformers huggingface_hub pillow accelerate

Usage:
  python qwen_ocr.py --image path/to/image.jpg
  python qwen_ocr.py --folder path/to/folder

Behavior:
- Downloads Qwen/Qwen2-VL-7B-Instruct once into models/Qwen2-VL-7B-Instruct.
- Loads from local files with offline mode enabled thereafter.
- Works with CPU or GPU (GPU recommended for speed and memory).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_DIR = Path("models") / "Qwen2-VL-7B-Instruct"
RESULTS_PATH = Path("results") / "qwen_results.jsonl"

SYSTEM_PROMPT = (
    "You are a medical prescription analyzer. Extract from this handwritten "
    "prescription image and return ONLY valid JSON with these fields: patient "
    "(name, age, gender), diagnosis, drugs (array with name, brand, dose, route, "
    "frequency, duration for each), comorbidities (array), pregnancy (true/false). "
    "Return only JSON, no explanation."
)

_PIPELINE: dict[str, Any] | None = None


def _set_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _unset_offline_mode() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


def _download_model_once() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))


def _load_local_pipeline() -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)

    return {
        "model": model,
        "processor": processor,
        "device": device,
    }


def _ensure_pipeline() -> dict[str, Any]:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    # Try fully offline local load first.
    _set_offline_mode()
    try:
        _PIPELINE = _load_local_pipeline()
        return _PIPELINE
    except Exception:
        pass

    # First-time setup path: go online once, download, then enforce offline.
    _unset_offline_mode()
    _download_model_once()
    _set_offline_mode()

    _PIPELINE = _load_local_pipeline()
    return _PIPELINE


def _extract_json(text: str) -> dict[str, Any]:
    # Try direct parse first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract first JSON object from free-form output.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")

    return json.loads(match.group(0))


def read_prescription(image_path: str) -> dict[str, Any]:
    """
    Reads a prescription image with local offline Qwen2-VL model.

    Returns parsed JSON on success.
    If parsing fails, returns:
      {"status": "error", "raw_text": "..."}
    """
    try:
        p = _ensure_pipeline()
        model = p["model"]
        processor = p["processor"]
        device = p["device"]

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Extract structured prescription JSON from this image.",
                    },
                ],
            },
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        # Move tensor inputs to device when running on CPU; for auto device map,
        # placing tensors on CUDA is still safe when available.
        if device == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=768)

        # Remove prompt tokens from generated output.
        trimmed_ids = []
        input_ids = inputs["input_ids"]
        for in_ids, out_ids in zip(input_ids, generated_ids):
            trimmed_ids.append(out_ids[len(in_ids) :])

        raw_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        try:
            parsed = _extract_json(raw_text)
            parsed["status"] = "success"
            return parsed
        except Exception:
            return {
                "status": "error",
                "raw_text": raw_text,
            }

    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "raw_text": "",
            "error": str(exc),
        }


def process_folder(folder_path: str) -> None:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder: {folder_path}")

    images = sorted(
        [
            p
            for p in folder.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        total = len(images)
        for idx, img in enumerate(images, start=1):
            result = read_prescription(str(img))
            result["image_path"] = str(img)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{idx}/{total}] processed: {img}")

    print(f"Saved results to {RESULTS_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Qwen2-VL prescription reader")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder of images for batch mode")
    args = parser.parse_args()

    if args.image and args.folder:
        print(json.dumps({"status": "error", "error": "Use either --image or --folder"}))
        return 1

    if args.image:
        result = read_prescription(args.image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "success" else 1

    if args.folder:
        process_folder(args.folder)
        return 0

    print(json.dumps({"status": "error", "error": "Provide --image or --folder"}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
