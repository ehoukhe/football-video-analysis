"""Orkestrerar hela analyskedjan: video -> detektering -> statistik -> output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from .annotate import draw_frame
from .detection import PlayerBallDetector
from .stats import MatchStats, generate_coach_insights
from .teams import AutoTeamClassifier, TeamClassifier
from .video import VideoReader, VideoWriter


@dataclass
class AnalysisResult:
    stats: MatchStats
    insights: list[str]
    output_video: Optional[str]
    heatmaps: list[str]
    stats_json: Optional[str]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_video(video_path: str, config: dict) -> AnalysisResult:
    """Kör hela analyspipelinen på en videofil enligt konfigurationen."""
    m = config["model"]
    v = config["video"]
    o = config["output"]
    t = config.get("teams", {})

    out_dir = Path(o.get("dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    detector = PlayerBallDetector(
        weights=m["weights"],
        conf=m["conf"],
        iou=m["iou"],
        device=m.get("device", ""),
        classes=m.get("classes"),
        tracker=config.get("tracker", {}).get("type", "bytetrack.yaml"),
        half=m.get("half", False),
        imgsz=m.get("imgsz", 640),
        player_conf=m.get("player_conf"),
        ball_conf=m.get("ball_conf"),
    )

    team_clf = None
    if t.get("enable"):
        if t.get("mode", "auto") == "auto":
            team_clf = AutoTeamClassifier(
                team_a_name=t.get("team_a_name", "Lag A"),
                team_b_name=t.get("team_b_name", "Lag B"),
            )
        else:
            team_clf = TeamClassifier(
                team_a_name=t["team_a_name"],
                team_b_name=t["team_b_name"],
                team_a_hsv_low=t["team_a_hsv_low"],
                team_a_hsv_high=t["team_a_hsv_high"],
                team_b_hsv_low=t["team_b_hsv_low"],
                team_b_hsv_high=t["team_b_hsv_high"],
            )

    writer: Optional[VideoWriter] = None
    heatmaps: list[str] = []
    stats_json: Optional[str] = None

    with VideoReader(
        video_path,
        resize_width=v.get("resize_width", 0),
        stride=v.get("frame_stride", 1),
        max_frames=v.get("max_frames", 0),
    ) as reader:
        meta = reader.meta
        stats = MatchStats(
            frame_width=meta.width,
            frame_height=meta.height,
            team_a_name=t.get("team_a_name", "Lag A"),
            team_b_name=t.get("team_b_name", "Lag B"),
        )

        if o.get("save_video"):
            out_video_path = str(out_dir / f"{stem}_annotated.mp4")
            writer = VideoWriter(
                out_video_path,
                fps=meta.fps / max(1, v.get("frame_stride", 1)),
                width=meta.width,
                height=meta.height,
            )
        else:
            out_video_path = None

        total = meta.frame_count if meta.frame_count > 0 else None
        for frame_idx, frame in tqdm(reader, total=total, desc="Analyserar", unit="ruta"):
            result = detector.track_frame(frame, frame_idx)

            if team_clf is not None:
                for det in result.players:
                    team_clf.classify(frame, det)

            stats.update(result)

            if writer is not None:
                writer.write(draw_frame(frame, result))

        if writer is not None:
            writer.release()

    # Heatmaps
    if o.get("save_heatmap"):
        bins = tuple(o.get("heatmap_bins", [68, 105]))
        p = stats.save_heatmap(str(out_dir / f"{stem}_heatmap_players.png"), "players", bins)
        if p:
            heatmaps.append(p)
        pb = stats.save_heatmap(str(out_dir / f"{stem}_heatmap_ball.png"), "ball", bins)
        if pb:
            heatmaps.append(pb)
        if team_clf is not None:
            names = (t.get("team_a_name", "Lag A"), t.get("team_b_name", "Lag B"))
            for team in names:
                pt = stats.save_heatmap(
                    str(out_dir / f"{stem}_heatmap_{team}.png"), "team", bins, team=team
                )
                if pt:
                    heatmaps.append(pt)

    # Statistik-JSON
    if o.get("save_stats"):
        stats_json = str(out_dir / f"{stem}_stats.json")
        stats.save_json(stats_json)

    insights = generate_coach_insights(stats)

    return AnalysisResult(
        stats=stats,
        insights=insights,
        output_video=out_video_path if writer is not None else None,
        heatmaps=heatmaps,
        stats_json=stats_json,
    )
