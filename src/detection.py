"""Detektering och spårning av spelare och boll med YOLOv8 (ultralytics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# COCO-klass-ID:n som YOLOv8 använder som standard.
PERSON_CLASS = 0
BALL_CLASS = 32


@dataclass
class Detection:
    """En enskild detektion i en bildruta."""

    frame_idx: int
    track_id: Optional[int]      # ID från trackern (None om spårning av)
    cls: int                     # COCO-klass (0 = person, 32 = boll)
    conf: float
    # Bounding box i pixlar: (x1, y1, x2, y2)
    xyxy: tuple[float, float, float, float]
    team: Optional[str] = None   # sätts av lag-klassificeraren

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Punkt vid fötterna (mitten av nederkanten) – bra för position på planen."""
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def is_ball(self) -> bool:
        return self.cls == BALL_CLASS

    @property
    def is_player(self) -> bool:
        return self.cls == PERSON_CLASS


@dataclass
class FrameResult:
    """Alla detektioner i en bildruta."""

    frame_idx: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def players(self) -> list[Detection]:
        return [d for d in self.detections if d.is_player]

    @property
    def balls(self) -> list[Detection]:
        return [d for d in self.detections if d.is_ball]


class PlayerBallDetector:
    """Wrapper runt en YOLOv8-modell med inbyggd spårning.

    Använder ultralytics ``model.track(...)`` som både detekterar och tilldelar
    stabila ``track_id`` över bildrutor (ByteTrack/BoT-SORT).
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.30,
        iou: float = 0.50,
        device: str = "",
        classes: Optional[list[int]] = None,
        tracker: str = "bytetrack.yaml",
        half: bool = False,
        imgsz: int = 640,
        player_conf: Optional[float] = None,
        ball_conf: Optional[float] = None,
    ) -> None:
        # Importeras lokalt för att hålla modulen importerbar utan tunga beroenden.
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.iou = iou
        self.device = device or None
        self.classes = classes if classes is not None else [PERSON_CLASS, BALL_CLASS]
        self.tracker = tracker
        # FP16 ger bara vinst på GPU; tvinga av på CPU för att undvika fel.
        self.half = bool(half) and self.device not in (None, "cpu")
        self.imgsz = imgsz
        # Separata trösklar: bollen är svårdetekterad (låg conf), spelare kan
        # ha högre tröskel för att sålla bort publik/bänk. Modellen körs på den
        # lägre av de två så inget missas, och filtreras sedan per klass.
        self.player_conf = player_conf if player_conf is not None else conf
        self.ball_conf = ball_conf if ball_conf is not None else conf
        self.conf = min(self.player_conf, self.ball_conf)

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """Kör detektering + spårning på en bildruta och returnerar resultatet.

        ``persist=True`` gör att trackern behåller tillstånd mellan anrop, vilket
        krävs när vi matar in en bildruta i taget.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            tracker=self.tracker,
            half=self.half,
            imgsz=self.imgsz,
            verbose=False,
        )

        result = FrameResult(frame_idx=frame_idx)
        if not results:
            return result

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return result

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), int)
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.full(len(xyxy), -1)
        )

        for box, conf, cls, tid in zip(xyxy, confs, clss, ids):
            # Filtrera per klass med respektive tröskel.
            if cls == BALL_CLASS and conf < self.ball_conf:
                continue
            if cls == PERSON_CLASS and conf < self.player_conf:
                continue
            result.detections.append(
                Detection(
                    frame_idx=frame_idx,
                    track_id=int(tid) if tid >= 0 else None,
                    cls=int(cls),
                    conf=float(conf),
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return result
