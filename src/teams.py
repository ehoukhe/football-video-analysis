"""Enkel lag-klassificering baserat på dominerande tröjfärg (HSV).

Detta är en heuristisk första version: den tittar på den övre delen av varje
spelares bounding box (ungefär tröjan) och avgör vilken av två färgprofiler den
bäst matchar. För en mer robust lösning kan man senare byta till t.ex.
k-means-klustring av färger eller en tränad klassificerare.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detection import Detection


class TeamClassifier:
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

    def _jersey_region(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        h = y2 - y1
        # Ta övre ~40 % av kroppen (tröjan), undvik huvud och ben.
        top = int(y1 + 0.15 * h)
        bottom = int(y1 + 0.55 * h)
        top = max(0, top)
        bottom = min(frame.shape[0], bottom)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        if bottom <= top or x2 <= x1:
            return None
        return frame[top:bottom, x1:x2]

    def classify(self, frame: np.ndarray, det: Detection) -> Optional[str]:
        """Returnerar lagnamn för en spelardetektion, eller None om osäkert."""
        if not det.is_player:
            return None
        region = self._jersey_region(frame, det)
        if region is None or region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask_a = cv2.inRange(hsv, self.a_low, self.a_high)
        mask_b = cv2.inRange(hsv, self.b_low, self.b_high)
        score_a = int(mask_a.sum())
        score_b = int(mask_b.sum())

        if score_a == 0 and score_b == 0:
            return None
        team = self.team_a_name if score_a >= score_b else self.team_b_name
        det.team = team
        return team
