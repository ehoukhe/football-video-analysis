#!/usr/bin/env python3
"""CLI för att analysera en fotbollsmatchvideo.

Exempel:

    # Analysera en lokal fil med standardkonfiguration
    python scripts/analyze_match.py --video data/match.mp4

    # Ladda ner från YouTube och analysera, snabbtest på 300 rutor
    python scripts/analyze_match.py --youtube "https://youtu.be/..." --max-frames 300

    # Använd en större/noggrannare modell
    python scripts/analyze_match.py --video data/match.mp4 --weights yolov8m.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Gör paketet 'src' importerbart oavsett var skriptet körs ifrån.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import analyze_video, load_config  # noqa: E402
from src.video import download_youtube  # noqa: E402


def _parse_time(value: str) -> float:
    """Tolkar '90', '1:30' eller '1:02:03' till sekunder."""
    parts = str(value).strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    secs = 0.0
    for part in parts:
        secs = secs * 60 + float(part)
    return secs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automatisk videoanalys av fotbollsmatcher.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Sökväg till lokal videofil.")
    src.add_argument("--youtube", help="YouTube-URL att ladda ner och analysera.")

    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Sökväg till YAML-konfiguration.",
    )
    p.add_argument("--weights", help="Överstyr YOLO-vikter, t.ex. yolov8m.pt.")
    p.add_argument("--conf", type=float, help="Överstyr konfidenströskel (0-1).")
    p.add_argument("--device", help="Överstyr enhet: '', 'cpu' eller '0'.")
    p.add_argument("--imgsz", type=int, help="Inferensupplösning (t.ex. 640, 960).")
    p.add_argument("--no-half", action="store_true", help="Stäng av FP16 (halvprecision).")
    p.add_argument("--resize-width", type=int, help="Överstyr nedskalningsbredd.")
    p.add_argument("--frame-stride", type=int, help="Läs var N:te bildruta.")
    p.add_argument("--max-frames", type=int, help="Max antal rutor (0 = alla).")
    p.add_argument("--start", help="Starta efter denna tid (sekunder eller mm:ss), skippar introt.")
    p.add_argument("--duration", help="Analysera bara detta segment (sekunder eller mm:ss).")
    p.add_argument("--output-dir", help="Överstyr output-katalog.")
    p.add_argument("--no-video", action="store_true", help="Spara inte annoterad video.")
    p.add_argument("--enable-teams", action="store_true", help="Aktivera lag-detektering.")
    return p.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.weights:
        config["model"]["weights"] = args.weights
    if args.conf is not None:
        config["model"]["conf"] = args.conf
    if args.device is not None:
        config["model"]["device"] = args.device
    if args.imgsz is not None:
        config["model"]["imgsz"] = args.imgsz
    if args.no_half:
        config["model"]["half"] = False
    if args.resize_width is not None:
        config["video"]["resize_width"] = args.resize_width
    if args.frame_stride is not None:
        config["video"]["frame_stride"] = args.frame_stride
    if args.max_frames is not None:
        config["video"]["max_frames"] = args.max_frames
    if args.start is not None:
        config["video"]["start_seconds"] = _parse_time(args.start)
    if args.duration is not None:
        config["video"]["duration_seconds"] = _parse_time(args.duration)
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    if args.no_video:
        config["output"]["save_video"] = False
    if args.enable_teams:
        config.setdefault("teams", {})["enable"] = True
    return config


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    if args.youtube:
        print(f"Laddar ner video från YouTube: {args.youtube}")
        video_path = download_youtube(args.youtube, out_dir="data")
        if not video_path:
            print("Kunde inte ladda ner videon.", file=sys.stderr)
            return 1
    else:
        video_path = args.video

    print(f"Analyserar: {video_path}")
    result = analyze_video(video_path, config)

    print("\n=== Sammanfattning ===")
    for key, value in result.stats.summary().items():
        print(f"  {key}: {value}")

    print("\n=== Insikter till tränaren ===")
    for i, insight in enumerate(result.insights, 1):
        print(f"  {i}. {insight}")

    print("\n=== Sparade filer ===")
    if result.output_video:
        print(f"  Annoterad video: {result.output_video}")
    for hm in result.heatmaps:
        print(f"  Heatmap: {hm}")
    if result.stats_json:
        print(f"  Statistik (JSON): {result.stats_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
