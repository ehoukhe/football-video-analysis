"""Plankalibrering: mappa bildpixlar till plankoordinater i meter (homografi).

Kärnan är :class:`ViewTransformer` som håller en 3x3-homografi och omvandlar
bildpunkter till planpunkter i meter. Homografin kan tas fram på två sätt:

* :class:`StaticCalibrator` – en fast homografi från punkt-korrespondenser.
  Fungerar för en STILLASTÅENDE taktikkamera (samma vy hela matchen).
* :class:`AutoKeypointCalibrator` – räknar om homografin per bildruta utifrån
  detekterade plan-nyckelpunkter. Krävs för en RÖRLIG kamera (XbotGo/sändning),
  men behöver en tränad nyckelpunktsmodell (Fas 2).

En kalibrator implementerar ``homography_for(frame, frame_idx) -> ViewTransformer | None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchModel:
    """Standardplan i meter. Origo i övre vänstra hörnet; x = längd, y = bredd."""

    length: float = 105.0
    width: float = 68.0

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Namngivna plankoordinater (meter) att para ihop med bildpunkter."""
        L, W = self.length, self.width
        pen_d = 16.5            # straffområdets djup
        pen_w = 40.32           # straffområdets bredd
        py0 = (W - pen_w) / 2.0
        py1 = (W + pen_w) / 2.0
        return {
            "corner_top_left": (0.0, 0.0),
            "corner_top_right": (L, 0.0),
            "corner_bottom_left": (0.0, W),
            "corner_bottom_right": (L, W),
            "center": (L / 2, W / 2),
            "center_line_top": (L / 2, 0.0),
            "center_line_bottom": (L / 2, W),
            "pen_left_top": (pen_d, py0),
            "pen_left_bottom": (pen_d, py1),
            "pen_right_top": (L - pen_d, py0),
            "pen_right_bottom": (L - pen_d, py1),
            "goal_left_top": (0.0, py0),
            "goal_left_bottom": (0.0, py1),
            "goal_right_top": (L, py0),
            "goal_right_bottom": (L, py1),
        }


class ViewTransformer:
    """Homografi bild -> plan (meter)."""

    def __init__(self, image_points, pitch_points) -> None:
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(pitch_points, dtype=np.float32)
        if len(src) < 4 or len(src) != len(dst):
            raise ValueError("Minst 4 punkt-par krävs (lika många i bild och plan).")
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
        if H is None:
            raise ValueError("Kunde inte beräkna homografi ur de givna punkterna.")
        self.H = H.astype(np.float64)

    @classmethod
    def from_matrix(cls, H) -> "ViewTransformer":
        obj = cls.__new__(cls)
        obj.H = np.asarray(H, dtype=np.float64)
        return obj

    def transform(self, points) -> np.ndarray:
        """Omvandla en lista bildpunkter [(x, y), ...] till planpunkter i meter."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)


class Calibrator(Protocol):
    def homography_for(
        self, frame: np.ndarray, frame_idx: int
    ) -> Optional[ViewTransformer]:
        ...


class StaticCalibrator:
    """En fast homografi för hela videon (stillastående kamera)."""

    def __init__(self, image_points, pitch_points) -> None:
        self._transformer = ViewTransformer(image_points, pitch_points)

    def homography_for(self, frame, frame_idx) -> Optional[ViewTransformer]:
        return self._transformer


class AutoKeypointCalibrator:
    """Per-ruta homografi via en plan-nyckelpunktsdetektor (Fas 2).

    ``detector`` är valfritt objekt som anropas ``detector(frame)`` och
    returnerar ett par ``(image_points, pitch_points_m)`` med de nyckelpunkter
    som hittades i rutan (redan konfidens-filtrerade och ihopparade), där
    ``image_points`` är i pixlar och ``pitch_points_m`` är motsvarande
    plankoordinater i meter. Returnera ``(None, None)`` eller för få punkter om
    rutan inte kan kalibreras. Homografin räknas om per ruta -> hanterar en
    rörlig kamera.

    Detektorn är modell-agnostisk: se :class:`YoloPitchKeypointDetector` för en
    ultralytics-baserad implementation.
    """

    def __init__(self, detector, min_points: int = 4) -> None:
        self.detector = detector
        self.min_points = min_points

    def homography_for(self, frame, frame_idx) -> Optional[ViewTransformer]:
        image_points, pitch_points = self.detector(frame)
        if image_points is None or len(image_points) < self.min_points:
            return None
        try:
            return ViewTransformer(image_points, pitch_points)
        except ValueError:
            return None


class YoloPitchKeypointDetector:
    """Plan-nyckelpunkter via en ultralytics YOLO-pose-modell.

    ``model`` är en laddad ``ultralytics.YOLO``-pose-modell som förutspår K
    plan-nyckelpunkter per bild. ``vertices_m`` är en (K, 2)-lista med varje
    nyckelpunkts kända plankoordinat i meter, i SAMMA ordning som modellens
    nyckelpunkter (hämtas t.ex. från Roboflows ``SoccerPitchConfiguration``).

    Anropas ``detector(frame) -> (image_points, pitch_points_m)`` och returnerar
    bara punkter med konfidens >= ``conf``.
    """

    def __init__(self, model, vertices_m, conf: float = 0.5, imgsz: int = 1280,
                 device: Optional[str] = None) -> None:
        self.model = model
        self.vertices = np.asarray(vertices_m, dtype=np.float32)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame):
        res = self.model.predict(
            frame, imgsz=self.imgsz, device=self.device, verbose=False
        )
        if not res:
            return None, None
        kps = res[0].keypoints
        if kps is None or kps.xy is None:
            return None, None
        xy = kps.xy.cpu().numpy()
        if xy.ndim == 3:
            xy = xy[0]  # första instansen (planet)
        conf = (
            kps.conf.cpu().numpy() if kps.conf is not None else np.ones(len(xy))
        )
        conf = conf[0] if conf.ndim == 2 else conf

        k = min(len(xy), len(self.vertices), len(conf))
        keep = np.where(conf[:k] >= self.conf)[0]
        if len(keep) < 4:
            return None, None
        return xy[keep], self.vertices[keep]


def draw_pitch_heatmap(
    points_m,
    path: str,
    pitch: Optional[PitchModel] = None,
    title: str = "Top-down heatmap (meter)",
    bins: tuple[int, int] = (68, 105),
) -> Optional[str]:
    """Ritar en top-down heatmap i plankoordinater med planens kontur."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return None

    pitch = pitch or PitchModel()
    L, W = pitch.length, pitch.width
    pts = np.asarray(points_m, dtype=float) if len(points_m) else np.zeros((0, 2))

    heat = np.zeros(bins)
    if len(pts):
        inside = (
            (pts[:, 0] >= 0) & (pts[:, 0] <= L) & (pts[:, 1] >= 0) & (pts[:, 1] <= W)
        )
        pin = pts[inside]
        if len(pin):
            heat, _, _ = np.histogram2d(
                pin[:, 1], pin[:, 0], bins=bins, range=[[0, W], [0, L]]
            )
            heat = gaussian_filter(heat, sigma=1.2)

    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.imshow(heat, cmap="hot", extent=[0, L, W, 0], aspect="auto", interpolation="bilinear")

    # Planens kontur
    line = dict(color="white", lw=1.5, fill=False)
    ax.add_patch(Rectangle((0, 0), L, W, **line))
    ax.plot([L / 2, L / 2], [0, W], color="white", lw=1.5)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **line))
    pen_w = 40.32
    ax.add_patch(Rectangle((0, (W - pen_w) / 2), 16.5, pen_w, **line))
    ax.add_patch(Rectangle((L - 16.5, (W - pen_w) / 2), 16.5, pen_w, **line))

    ax.set_xlim(0, L)
    ax.set_ylim(W, 0)
    ax.set_title(title)
    ax.set_xlabel("Längd (m)")
    ax.set_ylabel("Bredd (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
