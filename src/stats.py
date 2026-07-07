"""Statistik och visualisering: heatmaps och possession-estimering.

Samlar detektioner över hela matchen och producerar:
  * Heatmaps över var spelare (och bollen) befunnit sig.
  * En enkel possession-estimering: vilket lag som är närmast bollen räknas som
    "i bollinnehav" den bildrutan; andelen summeras över matchen.
  * Rörelse-/löpsträcka per spelar-ID (i pixlar, kan skalas till meter).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .detection import Detection, FrameResult


class MatchStats:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        team_a_name: str = "Lag A",
        team_b_name: str = "Lag B",
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name

        # Positioner (foot_point) för heatmaps.
        self.player_points: list[tuple[float, float]] = []
        self.ball_points: list[tuple[float, float]] = []
        self.team_points: dict[str, list[tuple[float, float]]] = defaultdict(list)

        # Bana per track_id, för löpsträcka.
        self.tracks: dict[int, list[tuple[float, float]]] = defaultdict(list)

        # Possession-räknare (antal bildrutor).
        self.possession_frames: dict[str, int] = defaultdict(int)
        self.total_possession_frames = 0

    def update(self, result: FrameResult) -> None:
        """Mata in en bildrutas detektioner."""
        for det in result.players:
            self.player_points.append(det.foot_point)
            if det.track_id is not None:
                self.tracks[det.track_id].append(det.foot_point)
            if det.team:
                self.team_points[det.team].append(det.foot_point)

        for det in result.balls:
            self.ball_points.append(det.center)

        self._update_possession(result)

    def _update_possession(self, result: FrameResult) -> None:
        balls = result.balls
        players = result.players
        if not balls or not players:
            return
        # Använd den bollen med högst konfidens.
        ball = max(balls, key=lambda b: b.conf)
        bx, by = ball.center

        nearest = min(
            players,
            key=lambda p: (p.center[0] - bx) ** 2 + (p.center[1] - by) ** 2,
        )
        if nearest.team:
            self.possession_frames[nearest.team] += 1
            self.total_possession_frames += 1

    # ------------------------------------------------------------------ #
    # Sammanställning
    # ------------------------------------------------------------------ #
    def possession_pct(self) -> dict[str, float]:
        if self.total_possession_frames == 0:
            return {}
        return {
            team: round(100.0 * n / self.total_possession_frames, 1)
            for team, n in self.possession_frames.items()
        }

    def distance_per_track(self) -> dict[int, float]:
        """Total löpsträcka (i pixlar) per spårat ID."""
        distances: dict[int, float] = {}
        for tid, pts in self.tracks.items():
            if len(pts) < 2:
                distances[tid] = 0.0
                continue
            arr = np.asarray(pts)
            steps = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            distances[tid] = float(steps.sum())
        return distances

    def summary(self) -> dict:
        return {
            "frames_with_players": len(self.player_points),
            "ball_detections": len(self.ball_points),
            "tracked_ids": len(self.tracks),
            "possession_pct": self.possession_pct(),
            "distance_px_per_track": {
                str(k): round(v, 1) for k, v in self.distance_per_track().items()
            },
        }

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Heatmaps
    # ------------------------------------------------------------------ #
    def _heatmap_grid(
        self, points: list[tuple[float, float]], bins: tuple[int, int]
    ) -> np.ndarray:
        if not points:
            return np.zeros(bins)
        arr = np.asarray(points)
        # x -> kolumn (planets längd), y -> rad (planets bredd)
        heat, _, _ = np.histogram2d(
            arr[:, 1],
            arr[:, 0],
            bins=bins,
            range=[[0, self.frame_height], [0, self.frame_width]],
        )
        return heat

    def save_heatmap(
        self,
        path: str,
        which: str = "players",
        bins: tuple[int, int] = (68, 105),
        team: Optional[str] = None,
    ) -> Optional[str]:
        """Sparar en heatmap som PNG. ``which`` = 'players' | 'ball' | 'team'."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return None

        if which == "ball":
            points = self.ball_points
            title = "Boll – heatmap"
        elif which == "team" and team is not None:
            points = self.team_points.get(team, [])
            title = f"{team} – heatmap"
        else:
            points = self.player_points
            title = "Spelare – heatmap"

        heat = self._heatmap_grid(points, bins)
        heat = gaussian_filter(heat, sigma=1.5)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        ax.imshow(heat, cmap="hot", interpolation="bilinear", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Planets längd →")
        ax.set_ylabel("Planets bredd →")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path


def generate_coach_insights(stats: MatchStats) -> list[str]:
    """Genererar enkla textbaserade insikter för tränaren utifrån statistiken."""
    insights: list[str] = []
    poss = stats.possession_pct()

    if poss:
        top_team = max(poss, key=poss.get)
        insights.append(
            f"Bollinnehav: {', '.join(f'{t} {p}%' for t, p in poss.items())}. "
            f"{top_team} dominerade innehavet."
        )
        # Balans-kommentar
        values = list(poss.values())
        if len(values) == 2 and abs(values[0] - values[1]) > 20:
            insights.append(
                "Stor skillnad i bollinnehav – överväg att jobba på att hålla bollen "
                "i laget med lägre innehav (pressresistens, passningsspel)."
            )
    else:
        insights.append(
            "Ingen possession-statistik kunde beräknas (aktivera lag-detektering i "
            "config för detta)."
        )

    dists = stats.distance_per_track()
    if dists:
        active = sorted(dists.items(), key=lambda kv: kv[1], reverse=True)[:3]
        insights.append(
            "Mest rörliga spelare (spår-ID, sträcka i pixlar): "
            + ", ".join(f"#{tid}: {d:.0f}" for tid, d in active)
        )

    if len(stats.ball_points) < 0.1 * max(1, len(stats.player_points)):
        insights.append(
            "Bollen detekterades sällan – prova en större YOLO-modell (yolov8m/l) "
            "eller högre videoupplösning för bättre bollspårning."
        )

    return insights
