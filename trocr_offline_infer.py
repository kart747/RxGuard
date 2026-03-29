#!/usr/bin/env python3
"""
One-file utility for TrOCR large handwritten model:
1) One-time online setup: install deps + download model files.
2) Fully offline inference from local model directory.

Examples:
  python trocr_offline_infer.py setup
  python trocr_offline_infer.py infer --image /path/to/image.jpg
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MODEL_ID = "microsoft/trocr-large-handwritten"
DEFAULT_MODEL_DIR = Path("./models/trocr-large-handwritten")


def run_cmd(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def install_dependencies() -> None:
    # Install only what is needed for setup + inference.
    packages = [
        "torch",
        "transformers>=4.40.0",
        "huggingface_hub>=0.20.0",
        "Pillow",
        "safetensors",
    ]
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


def download_model(model_dir: Path, model_id: str) -> None:
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} into {model_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # Force a local load once while online to verify everything exists.
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    TrOCRProcessor.from_pretrained(str(model_dir), local_files_only=True)
    VisionEncoderDecoderModel.from_pretrained(str(model_dir), local_files_only=True)

    print("Setup complete. You can now run offline inference.")


def run_offline_inference(model_dir: Path, image_path: Path) -> str:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Run setup first while online."
        )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from PIL import Image
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        str(model_dir), local_files_only=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install/download and run offline OCR with microsoft/trocr-large-handwritten."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="One-time online setup.")
    p_setup.add_argument("--model-id", default=MODEL_ID, help="HF model ID.")
    p_setup.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local model directory.",
    )
    p_setup.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip dependency installation.",
    )

    p_infer = sub.add_parser("infer", help="Run OCR fully offline.")
    p_infer.add_argument("--image", type=Path, required=True, help="Input image path.")
    p_infer.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local model directory from setup.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "setup":
            if not args.skip_install:
                install_dependencies()
            download_model(args.model_dir, args.model_id)
            return 0

        if args.command == "infer":
            text = run_offline_inference(args.model_dir, args.image)
            print("OCR:", text)
            return 0

        parser.error("Unknown command")
        return 2

    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
