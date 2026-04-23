#!/usr/bin/env python3
"""
scripts/test_api_cli.py
─────────────────────────────────────────────────────────────────────────────
Quick CLI smoke-test: send an image to the running API and print results.

Usage
─────
python scripts/test_api_cli.py --image /path/to/leaf.jpg
python scripts/test_api_cli.py --image /path/to/leaf.jpg --models efficientnet_b0
python scripts/test_api_cli.py --image /path/to/leaf.jpg --save_dir ./xai_output
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

API_BASE = os.getenv("CROPXAI_API", "http://localhost:8000")


def decode_and_save(b64_uri: str, path: Path):
    _, raw = b64_uri.split(",", 1)
    path.write_bytes(base64.b64decode(raw))


def main():
    p = argparse.ArgumentParser(description="CropXAI API smoke-test client")
    p.add_argument("--image",    type=Path,  required=True, help="Leaf image file")
    p.add_argument("--models",   type=str,   default=None,  help="Comma-separated model keys")
    p.add_argument("--save_dir", type=Path,  default=None,  help="Directory to save XAI images")
    p.add_argument("--api",      type=str,   default=API_BASE)
    args = p.parse_args()

    if not args.image.exists():
        print(f"✗ File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"🌐  API: {args.api}")
    print(f"🍃  Image: {args.image}  ({args.image.stat().st_size // 1024} KB)")

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{args.api}/health", timeout=5)
        health = r.json()
        print(f"✅  API healthy — models loaded: {health.get('models_loaded', [])}")
    except Exception as e:
        print(f"✗ Cannot reach API: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Predict ───────────────────────────────────────────────────────────────
    with open(args.image, "rb") as f:
        mime = "image/jpeg" if args.image.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        files = {"file": (args.image.name, f, mime)}
        data  = {"models": args.models} if args.models else {}
        print("\n⏳  Sending prediction request …")
        r = requests.post(f"{args.api}/api/v1/predict", files=files, data=data, timeout=180)

    if r.status_code != 200:
        print(f"✗ API error {r.status_code}: {r.text[:400]}", file=sys.stderr)
        sys.exit(1)

    result = r.json()

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS   ({len(result['model_results'])} models)")
    print(f"{'='*60}")

    for res in result["model_results"]:
        reliable_mark = " ★ RECOMMENDED" if res["reliable"] else ""
        pred = res["prediction"].replace("___", " — ").replace("_", " ")
        print(f"\n  {res['model_name']}{reliable_mark}")
        print(f"  {'─'*40}")
        print(f"  Prediction : {pred}")
        print(f"  Confidence : {res['confidence']*100:.2f}%")
        print(f"  Time       : {res['inference_ms']:.0f} ms")
        print(f"  Top-5:")
        for item in res["top5"][:5]:
            cls = item["class"].replace("___", " — ").replace("_", " ")
            bar = "█" * int(item["confidence"] * 20)
            print(f"    {cls:<45} {item['confidence']*100:5.1f}%  {bar}")

        if args.save_dir:
            out = args.save_dir / res["model_key"]
            out.mkdir(parents=True, exist_ok=True)
            decode_and_save(res["xai"]["grad_cam"], out / "gradcam.png")
            decode_and_save(res["xai"]["lime"],     out / "lime.png")
            decode_and_save(res["xai"]["ig"],       out / "ig.png")
            print(f"  💾  XAI images saved → {out}/")

    print(f"\n{'='*60}")
    print(f"Recommended model: {result['recommended_model']}")

    if args.save_dir:
        decode_and_save(result["original_image"], args.save_dir / "original.png")
        print(f"Original image saved → {args.save_dir}/original.png")


if __name__ == "__main__":
    main()
