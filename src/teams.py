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

from collections import defaultdict
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

    Robusthet mot de två vanliga felkällorna:

    * **Ljus/skugga:** färg-särdraget bygger på LAB-krominans (a*, b*) som är i
      stort sett oberoende av ljushet – samma tröja i sol och skugga ger samma
      färg.
    * **ID-fragmentering:** varje spår klassificeras på sin *medianfärg*, och
      klustercentran räknas om löpande från alla spårens medianfärger (inte bara
      de första rutorna). Samma spelare under nytt spår-ID får därför samma lag
      så länge tröjfärgen är stabil.
    """

    def __init__(
        self,
        team_a_name: str = "Lag A",
        team_b_name: str = "Lag B",
        min_box_height: int = 30,
        min_tracks_to_fit: int = 6,
        refit_interval: int = 200,
        **_ignore,
    ) -> None:
        self.team_names = (team_a_name, team_b_name)
        # Spelare vars box är lägre än så här (pixlar) är för små för att
        # tröjfärgen ska vara tillförlitlig -> tilldelas inget lag.
        self.min_box_height = min_box_height
        self.min_tracks_to_fit = min_tracks_to_fit
        self.refit_interval = refit_interval
        self._track_feats: dict[int, list[np.ndarray]] = defaultdict(list)
        self._centroids: Optional[np.ndarray] = None
        self._since_fit = 0

    @property
    def fitted(self) -> bool:
        return self._centroids is not None

    def _feature(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """Ljus-oberoende färg-särdrag: median (a*, b*) i LAB över tröjpixlar.

        Gräsgröna, mycket mörka och mycket ljusa pixlar maskas bort. a*/b* är
        krominans (färg) skild från L* (ljushet), så sol/skugga påverkar knappt.
        """
        region = _jersey_region(frame, det)
        if region is None or region.size == 0:
            return None
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.int16)
        v = hsv[:, :, 2].astype(np.int16)

        # Ignorera gräsgröna (hue ~35-85 i OpenCV) samt nära svart/vitt.
        keep = (v > 40) & (v < 250) & ~((h >= 35) & (h <= 85))
        if int(keep.sum()) < 20:
            keep = (v > 20) & (v < 252)
        if int(keep.sum()) < 5:
            return None

        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        # Skala till ~[-1, 1] (128 = neutral).
        feat = np.array(
            [(float(np.median(a[keep])) - 128.0) / 128.0,
             (float(np.median(b[keep])) - 128.0) / 128.0],
            dtype=np.float32,
        )
        return feat

    def _track_color(self, tid: int) -> np.ndarray:
        return np.median(np.vstack(self._track_feats[tid]), axis=0).astype(np.float32)

    def _maybe_fit(self) -> None:
        # Efter första anpassningen: räkna bara om ibland (spar tid).
        if self._centroids is not None and self._since_fit < self.refit_interval:
            return
        medians = [
            np.median(np.vstack(f), axis=0)
            for f in self._track_feats.values()
            if len(f) >= 2
        ]
        if len(medians) < self.min_tracks_to_fit:
            return
        data = np.vstack(medians).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, _, centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        self._centroids = centers
        self._since_fit = 0

    def _assign(self, feat: np.ndarray) -> str:
        dists = np.linalg.norm(self._centroids - feat, axis=1)
        return self.team_names[int(np.argmin(dists))]

    def classify(self, frame: np.ndarray, det: Detection) -> Optional[str]:
        if not det.is_player:
            return None
        # Hoppa över för små spelare – deras tröjfärg är opålitlig.
        x1, y1, x2, y2 = det.xyxy
        if (y2 - y1) < self.min_box_height:
            return None
        if det.track_id is None:
            return None  # kräver spår-ID för stabil färg per spår

        feat = self._feature(frame, det)
        if feat is None:
            return None

        self._track_feats[det.track_id].append(feat)
        self._since_fit += 1
        self._maybe_fit()
        if not self.fitted:
            return None  # ännu inte tillräckligt underlag för klustring

        team = self._assign(self._track_color(det.track_id))
        det.team = team
        return team
