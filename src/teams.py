"""Lag-klassificering baserat på tröjfärg.

Två varianter:

* :class:`TeamClassifier` – manuell: du anger två HSV-färgintervall och varje
  spelare tilldelas det lag vars färg dominerar i tröjregionen.
* :class:`AutoTeamClassifier` – automatisk: samlar tröjfärger under ett antal
  "uppvärmningsrutor", klustrar dem i två grupper (k-means) och tilldelar sedan
  spelare till närmaste lagfärg. Kräver ingen handinställning av färger.

Båda tittar bara på den övre delen av bounding-boxen (ungefär tröjan) och
ignorerar gräsgröna/omättade pixlar så att planen inte stör färgbedömningen.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

import cv2
import numpy as np

from .detection import Detection


def _jersey_region(frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
    """Klipper ut tröjregionen (övre delen av kroppen) ur bildrutan."""
    x1, y1, x2, y2 = (int(v) for v in det.xyxy)
    h = y2 - y1
    top = max(0, int(y1 + 0.15 * h))
    bottom = min(frame.shape[0], int(y1 + 0.55 * h))
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)
    if bottom <= top or x2 <= x1:
        return None
    return frame[top:bottom, x1:x2]


class TeamClassifier:
    """Manuell klassificering via två fasta HSV-färgintervall."""

    def __init__(
        self,
        team_a_name: str,
        team_b_name: str,
        team_a_hsv_low: list[int],
        team_a_hsv_high: list[int],
        team_b_hsv_low: list[int],
        team_b_hsv_high: list[int],
    ) -> None:
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.a_low = np.array(team_a_hsv_low, dtype=np.uint8)
        self.a_high = np.array(team_a_hsv_high, dtype=np.uint8)
        self.b_low = np.array(team_b_hsv_low, dtype=np.uint8)
        self.b_high = np.array(team_b_hsv_high, dtype=np.uint8)

    def classify(self, frame: np.ndarray, det: Detection) -> Optional[str]:
        if not det.is_player:
            return None
        region = _jersey_region(frame, det)
        if region is None or region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        score_a = int(cv2.inRange(hsv, self.a_low, self.a_high).sum())
        score_b = int(cv2.inRange(hsv, self.b_low, self.b_high).sum())
        if score_a == 0 and score_b == 0:
            return None
        team = self.team_a_name if score_a >= score_b else self.team_b_name
        det.team = team
        return team


class AutoTeamClassifier:
    """Automatisk lag-klassificering via k-means på tröjfärg.

    Under de första rutorna samlas färg-särdrag in. När tillräckligt många
    exempel finns anpassas två klustercentra (k-means) en gång; därefter
    tilldelas varje spelare till närmaste kluster. Innan modellen anpassats
    returneras ``None`` (possession räknas alltså först när lagen är kända).
    """

    def __init__(
        self,
        team_a_name: str = "Lag A",
        team_b_name: str = "Lag B",
        sample_target: int = 300,
        min_box_height: int = 30,
    ) -> None:
        self.team_names = (team_a_name, team_b_name)
        self.sample_target = sample_target
        # Spelare vars box är lägre än så här (pixlar) är för små för att
        # tröjfärgen ska vara tillförlitlig -> tilldelas inget lag.
        self.min_box_height = min_box_height
        self._samples: list[np.ndarray] = []
        self._centroids: Optional[np.ndarray] = None
        # Röster per spår-ID -> stabil lagtillhörighet (spelare byter inte lag
        # mellan rutor även om enskilda mätningar spretar).
        self._track_votes: dict[int, Counter] = defaultdict(Counter)

    @property
    def fitted(self) -> bool:
        return self._centroids is not None

    def _feature(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """Robust färg-särdrag för tröjan: hue som (cos, sin) + mättnad + ljushet.

        Hue kodas cirkulärt så att röd (~0) och röd (~179) ligger nära varandra.
        S/V gör att vita/svarta tröjor (låg mättnad) ändå kan skiljas åt.
        """
        region = _jersey_region(frame, det)
        if region is None or region.size == 0:
            return None
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        # Ignorera gräsgröna (hue ~35-85 i OpenCV) och mycket mörka pixlar.
        keep = (v > 40) & ~((h >= 35) & (h <= 85))
        if int(keep.sum()) < 20:
            keep = v > 20  # nödfall: använd allt utom nästan svart
        if int(keep.sum()) < 5:
            return None

        hue_rad = h[keep] * (2.0 * math.pi / 180.0)
        feat = np.array(
            [
                float(np.mean(np.cos(hue_rad))),
                float(np.mean(np.sin(hue_rad))),
                float(np.mean(s[keep]) / 255.0),
                float(np.mean(v[keep]) / 255.0),
            ],
            dtype=np.float32,
        )
        return feat

    def _fit(self) -> None:
        data = np.vstack(self._samples).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, _, centers = cv2.kmeans(
            data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        self._centroids = centers  # shape (2, 4)

    def classify(self, frame: np.ndarray, det: Detection) -> Optional[str]:
        if not det.is_player:
            return None
        # Hoppa över för små spelare – deras tröjfärg är opålitlig.
        x1, y1, x2, y2 = det.xyxy
        if (y2 - y1) < self.min_box_height:
            return None

        feat = self._feature(frame, det)
        if feat is None:
            return None

        if not self.fitted:
            self._samples.append(feat)
            if len(self._samples) >= self.sample_target:
                self._fit()
            return None  # lagen ännu inte bestämda

        dists = np.linalg.norm(self._centroids - feat, axis=1)
        inst_team = self.team_names[int(np.argmin(dists))]

        # Utan spår-ID: använd den momentana bedömningen.
        if det.track_id is None:
            det.team = inst_team
            return inst_team

        # Med spår-ID: rösta och returnera majoriteten för det spåret (stabilt).
        votes = self._track_votes[det.track_id]
        votes[inst_team] += 1
        team = votes.most_common(1)[0][0]
        det.team = team
        return team
