"""Hjälpfunktioner för att läsa och skriva videofiler med OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class VideoMeta:
    """Metadata om en videofil."""

    fps: float
    width: int
    height: int
    frame_count: int
    path: str


class VideoReader:
    """Läser bildrutor från en video, med valfri nedskalning och stride.

    Används som kontexthanterare:

        with VideoReader("match.mp4", resize_width=1280, stride=2) as reader:
            for frame_idx, frame in reader:
                ...
    """

    def __init__(
        self,
        path: str,
        resize_width: int = 0,
        stride: int = 1,
        max_frames: int = 0,
        start_seconds: float = 0.0,
        duration_seconds: float = 0.0,
    ) -> None:
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Videofilen hittades inte: {self.path}")

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Kunde inte öppna videon: {self.path}")

        self.resize_width = resize_width
        self.stride = max(1, stride)
        self.max_frames = max_frames

        src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tidsfönster: hoppa över introt och/eller analysera bara ett segment.
        self.start_frame = max(0, int(round(start_seconds * fps)))
        self.end_frame = (
            self.start_frame + int(round(duration_seconds * fps))
            if duration_seconds and duration_seconds > 0
            else None
        )

        # Beräkna utdata-dimensioner efter ev. skalning.
        if resize_width and src_w > 0 and resize_width < src_w:
            scale = resize_width / src_w
            out_w = resize_width
            out_h = int(round(src_h * scale))
        else:
            out_w, out_h = src_w, src_h

        self.meta = VideoMeta(
            fps=fps,
            width=out_w,
            height=out_h,
            frame_count=count,
            path=self.path,
        )

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        if self.meta.width and frame.shape[1] != self.meta.width:
            return cv2.resize(frame, (self.meta.width, self.meta.height))
        return frame

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        # Sök fram till startrutan (hoppar över introt).
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        idx = self.start_frame - 1
        emitted = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            idx += 1
            if self.end_frame is not None and idx >= self.end_frame:
                break
            # Räkna stride relativt startrutan så första rutan alltid tas med.
            if (idx - self.start_frame) % self.stride != 0:
                continue
            yield idx, self._maybe_resize(frame)
            emitted += 1
            if self.max_frames and emitted >= self.max_frames:
                break

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoWriter:
    """Skriver annoterade bildrutor till en MP4-fil."""

    def __init__(self, path: str, fps: float, width: int, height: int) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise IOError(f"Kunde inte skapa videoskrivare: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def download_youtube(url: str, out_dir: str = "data") -> Optional[str]:
    """Laddar ner en YouTube-video med yt-dlp och returnerar filsökvägen.

    Kräver att paketet ``yt-dlp`` är installerat. Returnerar None vid fel.
    """
    try:
        import yt_dlp  # importeras lokalt så det är valfritt
    except ImportError:
        raise ImportError(
            "yt-dlp krävs för YouTube-nedladdning. Kör: pip install yt-dlp"
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio/best[ext=mp4]/best",
        "outtmpl": str(Path(out_dir) / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)
