"""Ritar bounding boxes, ID:n och lag-etiketter på bildrutor."""

from __future__ import annotations

import cv2
import numpy as np

from .detection import FrameResult

# Färger (BGR)
COLOR_BALL = (0, 255, 255)      # gul
COLOR_PLAYER = (0, 255, 0)      # grön
TEAM_COLORS = {
    "A": (255, 128, 0),         # blå-ish
    "B": (0, 0, 255),           # röd
}


def _team_color(team: str | None) -> tuple[int, int, int]:
    if not team:
        return COLOR_PLAYER
    # Deterministisk färg per lagnamn.
    key = "A" if hash(team) % 2 == 0 else "B"
    return TEAM_COLORS[key]


def draw_frame(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Returnerar en kopia av bildrutan med annoteringar."""
    out = frame.copy()

    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        if det.is_ball:
            color = COLOR_BALL
            label = f"boll {det.conf:.2f}"
            cv2.circle(out, det.center_int, 8, color, 2)
        else:
            color = _team_color(det.team)
            parts = []
            if det.track_id is not None:
                parts.append(f"#{det.track_id}")
            if det.team:
                parts.append(det.team)
            label = " ".join(parts) if parts else "spelare"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _draw_label(out, label, x1, y1, color)

    return out


def _draw_label(frame, text, x, y, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y_top = max(0, y - th - 6)
    cv2.rectangle(frame, (x, y_top), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 4), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


# Bekvämlighetsegenskap på Detection utan att röra dataklassen:
def _center_int(self) -> tuple[int, int]:
    cx, cy = self.center
    return int(cx), int(cy)


from .detection import Detection  # noqa: E402

Detection.center_int = property(_center_int)
