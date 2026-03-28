#!/usr/bin/env python3
"""
Handwritten medical prescription OCR using PaddleOCR.

Install requirements:
  pip install --upgrade paddleocr paddlepaddle opencv-python numpy

Usage:
  python script.py image.jpg
  python script.py --folder /path/to/images --jsonl results/paddle_results.jsonl

Notes:
- Uses PaddleOCR with use_angle_cls=True, lang='en'.
- Keeps the same OpenCV preprocessing pipeline as requested.
"""

from __future__ import annotations

import os
# Disable MKL-DNN (oneDNN) backend — causes ConvertPirAttribute crash on CPU.
# Skip HuggingFace connectivity check — models already cached.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from drug_normalization import normalize_drug_list
from med7_ner import extract_med7_entities


# Set by run_ocr for CLI save behavior.
_LAST_PREPROCESSED: Optional[np.ndarray] = None
_OCR_ENGINE: Optional[Any] = None
_MAX_OCR_SIDE = int(os.getenv("OCR_MAX_SIDE", "2048"))
_DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.80"))
_RXNORM_ENABLED = os.getenv("RXNORM_LOOKUP", "1") != "0"
_RXNORM_TIMEOUT_SEC = float(os.getenv("RXNORM_TIMEOUT_SEC", "3.0"))
_MED7_ENABLED = os.getenv("MED7_ENABLED", "1") != "0"
_MED7_MODEL = os.getenv("MED7_MODEL", "en_core_web_sm")


def _normalize_medical_text(text: str) -> tuple[str, list[str]]:
    if not text:
        return "", []

    normalized = text
    actions: list[str] = []

    # Normalize common medical shorthand first.
    abbreviations: list[tuple[str, str, str]] = [
        (r"\b(?:b\.?d\.?|bd|bid)\b", "BID", "freq:BD/BID->BID"),
        (r"\b(?:o\.?d\.?|od|qd)\b", "QD", "freq:OD/QD->QD"),
        (r"\b(?:t\.?i\.?d\.?|tid)\b", "TID", "freq:TID->TID"),
        (r"\b(?:q\.?i\.?d\.?|qid)\b", "QID", "freq:QID->QID"),
        (r"\b(?:h\.?s\.?|hs)\b", "HS", "freq:HS->HS"),
        (r"\bs\.?o\.?s\.?\b", "SOS", "freq:SOS->SOS"),
        (r"\b(?:p\.?o\.?|po)\b", "PO", "route:PO->PO"),
    ]
    for pattern, replacement, action in abbreviations:
        updated = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        if updated != normalized:
            normalized = updated
            actions.append(action)

    # Normalize dose units and spacing.
    unit_patterns: list[tuple[str, str, str]] = [
        (r"\b(\d+(?:\.\d+)?)\s*(?:mgm|mgs|mg\.)\b", r"\1 mg", "unit:mg"),
        (r"\b(\d+(?:\.\d+)?)\s*(?:ml\.|m\.l\.)\b", r"\1 ml", "unit:ml"),
        (r"\b(\d+(?:\.\d+)?)\s*(?:mcg|ug|μg)\b", r"\1 mcg", "unit:mcg"),
        (r"\b(\d+(?:\.\d+)?)\s*(?:gm|gms|g\.)\b", r"\1 g", "unit:g"),
    ]
    for pattern, replacement, action in unit_patterns:
        updated = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        if updated != normalized:
            normalized = updated
            actions.append(action)

    collapsed = re.sub(r"\s+", " ", normalized).strip()
    if collapsed != normalized:
        actions.append("whitespace:collapse")
    normalized = collapsed

    return normalized, sorted(set(actions))


def _review_reasons(
    *,
    status: str,
    confidence: float,
    raw_text: str,
    review_policy: str,
    confidence_threshold: float,
) -> list[str]:
    reasons: list[str] = []

    if review_policy == "all":
        reasons.append("phase1_all_review_policy")

    if status != "success":
        reasons.append("ocr_error")

    if confidence < confidence_threshold:
        reasons.append("low_confidence")

    if len(raw_text.strip()) < 3:
        reasons.append("empty_or_short_text")

    return reasons


def _extract_rx_text_block(corrected_text: str) -> str:
    if not corrected_text:
        return ""

    # Phase 2 region-first scaffold: isolate likely medication segment.
    match = re.search(
        r"\bRx\b[:,\-]?\s*(.+?)(?:\bAdv\b|\bAdvice\b|$)",
        corrected_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return corrected_text
    return match.group(1).strip()


def _ensure_ocr() -> Any:
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    from paddleocr import PaddleOCR

    # enable_mkldnn=False: disables oneDNN backend which causes
    # a ConvertPirAttribute2RuntimeAttribute crash on CPU inference.
    _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)
    return _OCR_ENGINE


def _read_image_bgr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return img


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    # Use foreground mask for minAreaRect angle estimation.
    fg = 255 - binary_img
    coords = np.column_stack(np.where(fg > 0))
    if coords.size == 0:
        return binary_img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_img,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(image_path: str, bgr: Optional[np.ndarray] = None) -> np.ndarray:
    if bgr is None:
        bgr = _read_image_bgr(image_path)

    # Required preprocessing pipeline.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoise = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)  # h=10: lighter denoising preserves pen strokes
    thresh = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    deskewed = _deskew(thresh)
    equalized = cv2.equalizeHist(deskewed)
    kernel = np.ones((1, 1), np.uint8)
    closed = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)

    return closed


def _prepare_ocr_input(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= _MAX_OCR_SIDE:
        return bgr

    scale = _MAX_OCR_SIDE / float(max_side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _ocr_with_fallback(ocr: Any, bgr: np.ndarray) -> Any:
    ocr_input = _prepare_ocr_input(bgr)
    try:
        return ocr.ocr(ocr_input)
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        memory_signals = ("memory", "bad_alloc", "std::bad_alloc", "out of memory")
        if not any(sig in message for sig in memory_signals):
            raise

    # Fallback for memory-heavy images: process in vertical tiles.
    h, w = bgr.shape[:2]
    tiles = min(4, max(2, int(np.ceil(h / 1200))))
    step = int(np.ceil(h / float(tiles)))
    merged: list[Any] = []

    for i in range(tiles):
        start = i * step
        end = min(h, (i + 1) * step)
        if end <= start:
            continue
        tile = bgr[start:end, 0:w]
        tile_input = _prepare_ocr_input(tile)
        tile_result = ocr.ocr(tile_input)
        if isinstance(tile_result, list):
            merged.extend(tile_result)
        elif tile_result is not None:
            merged.append(tile_result)

    return merged


def _extract_text_and_confidence(ocr_result: Any) -> tuple[str, float]:
    texts: list[str] = []
    scores: list[float] = []

    if not ocr_result:
        return "", 0.0

    # ── PaddleOCR v3: returns a list of OCRResult objects (or a single one) ──
    # Each OCRResult has 'rec_texts' and 'rec_scores' attributes/keys.
    items = ocr_result if isinstance(ocr_result, list) else [ocr_result]
    for item in items:
        # v3 dict/object with rec_texts key
        rec_texts = None
        rec_scores = None
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts")
            rec_scores = item.get("rec_scores")
        elif hasattr(item, "rec_texts"):
            rec_texts = item.rec_texts
            rec_scores = getattr(item, "rec_scores", None)

        if rec_texts is not None:
            for i, txt in enumerate(rec_texts):
                txt = str(txt).strip()
                if txt:
                    texts.append(txt)
                    if rec_scores is not None and i < len(rec_scores):
                        try:
                            scores.append(float(rec_scores[i]))
                        except Exception:  # noqa: BLE001
                            scores.append(0.0)
            continue  # handled this item

        # ── PaddleOCR v2 legacy: nested list [ [[box], [text, score]], ... ] ──
        lines = item[0] if isinstance(item, list) and len(item) > 0 else item
        if lines is None:
            continue
        for line in lines:
            if not line:
                continue
            rec = None
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                rec = line[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                txt = str(rec[0]).strip()
                try:
                    sc = float(rec[1])
                except Exception:  # noqa: BLE001
                    sc = 0.0
                if txt:
                    texts.append(txt)
                    scores.append(sc)
            elif isinstance(rec, str):
                txt = rec.strip()
                if txt:
                    texts.append(txt)

    raw_text = " ".join(texts).strip()
    confidence = float(np.mean(scores)) if scores else 0.0
    confidence = max(0.0, min(1.0, confidence))
    return raw_text, confidence


def run_ocr(
    image_path: str,
    ground_truth: Optional[str] = None,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    review_policy: str = "all",
) -> dict[str, Any]:
    """
    Runs OCR and returns a JSON-serializable dictionary.

    Returns:
      {
        "raw_text": str,
        "corrected_text": str,
        "confidence": float,
        "status": "success" | "error"
      }
    """
    del ground_truth  # kept only to preserve the same signature

    global _LAST_PREPROCESSED

    result: dict[str, Any] = {
        "raw_text": "",
        "corrected_text": "",
        "phase2_rx_text": "",
        "normalized_drugs": {"drugs": [], "meta": {"count": 0, "rxnorm_enabled": _RXNORM_ENABLED}},
        "module3_ner": {
            "status": "unavailable",
            "model": _MED7_MODEL,
            "error": "not_run",
            "patient": {"name": None, "age": None, "gender": None},
            "diagnosis": None,
            "entities": [],
            "drug_entities": [],
        },
        "confidence": 0.0,
        "normalization_actions": [],
        "needs_review": True,
        "review_reasons": ["phase1_all_review_policy"],
        "status": "error",
    }

    try:
        ocr = _ensure_ocr()
        bgr = _read_image_bgr(image_path)

        # Save preprocessed image for reference but DON'T feed it to PaddleOCR.
        # PaddleOCR has its own internal preprocessing pipeline optimised for its
        # detection network — binarized images break it completely.
        processed = preprocess_image(image_path, bgr=bgr)
        _LAST_PREPROCESSED = processed.copy()

        # Feed the original-color image as an array, resized to cap memory use.
        # Note: In PaddleOCR v3, cls=True is no longer a valid kwarg.
        ocr_result = _ocr_with_fallback(ocr, bgr)
        raw_text, confidence = _extract_text_and_confidence(ocr_result)

        # Keep corrected_text field for compatibility with prior outputs.
        corrected_text, normalization_actions = _normalize_medical_text(raw_text)
        phase2_rx_text = _extract_rx_text_block(corrected_text)
        normalized_drugs = normalize_drug_list(
            corrected_text=phase2_rx_text,
            use_rxnorm=_RXNORM_ENABLED,
            timeout_sec=_RXNORM_TIMEOUT_SEC,
        )
        module3_ner = (
            extract_med7_entities(corrected_text, model_name=_MED7_MODEL)
            if _MED7_ENABLED
            else {
                "status": "disabled",
                "model": _MED7_MODEL,
                "error": "med7_disabled_by_flag",
                "patient": {"name": None, "age": None, "gender": None},
                "diagnosis": None,
                "entities": [],
                "drug_entities": [],
            }
        )
        review_reasons = _review_reasons(
            status="success",
            confidence=confidence,
            raw_text=raw_text,
            review_policy=review_policy,
            confidence_threshold=confidence_threshold,
        )

        result.update(
            {
                "raw_text": raw_text,
                "corrected_text": corrected_text,
                "phase2_rx_text": phase2_rx_text,
                "normalized_drugs": normalized_drugs,
                "module3_ner": module3_ner,
                "confidence": confidence,
                "normalization_actions": normalization_actions,
                "needs_review": len(review_reasons) > 0,
                "review_reasons": review_reasons,
                "status": "success",
            }
        )
        return result

    except Exception as exc:  # noqa: BLE001
        review_reasons = _review_reasons(
            status="error",
            confidence=0.0,
            raw_text="",
            review_policy=review_policy,
            confidence_threshold=confidence_threshold,
        )
        result["status"] = "error"
        result["needs_review"] = True
        result["review_reasons"] = review_reasons
        result["error"] = str(exc)
        return result


def _image_files(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def _save_corrected_image(out_path: Path) -> None:
    if _LAST_PREPROCESSED is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), _LAST_PREPROCESSED)


def _run_single(
    image_path: str,
    _ground_truth_path: Optional[str],
    confidence_threshold: float,
    review_policy: str,
) -> int:
    result = run_ocr(
        image_path,
        confidence_threshold=confidence_threshold,
        review_policy=review_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    corrected_path = Path("corrected.jpg")
    _save_corrected_image(corrected_path)
    return 0 if result.get("status") == "success" else 1


def _run_batch(
    folder: str,
    jsonl_out: str,
    save_corrected_dir: Optional[str],
    confidence_threshold: float,
    review_policy: str,
    review_queue_out: Optional[str],
) -> int:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(json.dumps({"status": "error", "error": f"Invalid folder: {folder}"}))
        return 1

    images = _image_files(folder_path)
    out_path = Path(jsonl_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_paths: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img_path = item.get("image_path")
                if isinstance(img_path, str) and img_path:
                    done_paths.add(img_path)

    pending_images = [img for img in images if str(img) not in done_paths]

    corrected_root = Path(save_corrected_dir) if save_corrected_dir else None
    review_queue_path = Path(review_queue_out) if review_queue_out else None
    if review_queue_path is not None:
        review_queue_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    review_count = 0
    write_mode = "a" if done_paths else "w"
    with out_path.open(write_mode, encoding="utf-8") as f:
        review_file = (
            review_queue_path.open("a", encoding="utf-8")
            if review_queue_path is not None
            else None
        )
        try:
            for img in pending_images:
                item = run_ocr(
                    str(img),
                    confidence_threshold=confidence_threshold,
                    review_policy=review_policy,
                )
                item["image_path"] = str(img)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()

                if item.get("needs_review") and review_file is not None:
                    review_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    review_file.flush()
                    review_count += 1

                if corrected_root is not None:
                    rel = img.relative_to(folder_path)
                    corrected_path = corrected_root / rel
                    corrected_path = corrected_path.with_suffix(".jpg")
                    _save_corrected_image(corrected_path)

                processed_count += 1
        finally:
            if review_file is not None:
                review_file.close()

    summary = {
        "status": "success",
        "jsonl": str(out_path),
        "processed": processed_count,
        "review_items": review_count,
        "skipped_existing": len(done_paths),
        "total_images": len(images),
    }
    if review_queue_path is not None:
        summary["review_queue"] = str(review_queue_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handwritten medical prescription OCR")
    parser.add_argument("image", nargs="?", help="Input image path for single-image mode")
    parser.add_argument(
        "ground_truth",
        nargs="?",
        help="Unused in PaddleOCR mode (kept for API compatibility)",
    )
    parser.add_argument("--folder", help="Batch folder path containing images")
    parser.add_argument(
        "--jsonl",
        default="results/paddle_results.jsonl",
        help="Output JSONL path for batch mode",
    )
    parser.add_argument(
        "--save-corrected",
        default=None,
        help="Optional folder to save preprocessed images in batch mode",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=_DEFAULT_CONFIDENCE_THRESHOLD,
        help="Mark results below this confidence for review",
    )
    parser.add_argument(
        "--review-policy",
        choices=["all", "low-confidence"],
        default="all",
        help="Phase-1 default is all prescriptions routed for review",
    )
    parser.add_argument(
        "--review-queue",
        default="results/review_queue.jsonl",
        help="JSONL output for items flagged for review (set empty string to disable)",
    )
    parser.add_argument(
        "--disable-rxnorm",
        action="store_true",
        help="Disable online RxNorm lookup and use local normalization only",
    )
    parser.add_argument(
        "--rxnorm-timeout",
        type=float,
        default=_RXNORM_TIMEOUT_SEC,
        help="Timeout in seconds for each RxNorm API request",
    )
    parser.add_argument(
        "--disable-med7",
        action="store_true",
        help="Disable Module 3 NER extraction",
    )
    parser.add_argument(
        "--med7-model",
        type=str,
        default=_MED7_MODEL,
        help="spaCy model name for Module 3 extractor (default: en_core_web_sm)",
    )
    return parser.parse_args()


def main() -> int:
    global _RXNORM_ENABLED, _RXNORM_TIMEOUT_SEC, _MED7_ENABLED, _MED7_MODEL

    args = parse_args()
    if args.disable_rxnorm:
        _RXNORM_ENABLED = False
    _RXNORM_TIMEOUT_SEC = float(args.rxnorm_timeout)
    if args.disable_med7:
        _MED7_ENABLED = False
    _MED7_MODEL = args.med7_model

    if args.folder:
        review_queue_out = args.review_queue.strip() if args.review_queue else None
        return _run_batch(
            args.folder,
            args.jsonl,
            args.save_corrected,
            args.confidence_threshold,
            args.review_policy,
            review_queue_out,
        )

    if not args.image:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "Provide an image path or use --folder for batch mode.",
                },
                ensure_ascii=False,
            )
        )
        return 1

    return _run_single(
        args.image,
        args.ground_truth,
        args.confidence_threshold,
        args.review_policy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
