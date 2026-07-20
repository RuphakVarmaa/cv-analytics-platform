import asyncio
import time
from typing import AsyncGenerator, Optional
import cv2
import numpy as np

from app.config import settings


class FrameCaptureError(Exception):
    """Raised when frame capture cannot be initialized or fails."""
    pass


def _letterbox_frame(frame: np.ndarray, max_size: int = 640) -> np.ndarray:
    """
    Resize frame so the longest side is at most max_size,
    preserving aspect ratio (letterbox padding not applied — pure resize).
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR numpy frame to JPEG bytes."""
    frame = _letterbox_frame(frame)
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise FrameCaptureError("JPEG encoding failed")
    return buf.tobytes()


def _open_capture(path_or_index) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path_or_index)
    if not cap.isOpened():
        raise FrameCaptureError(f"Cannot open video source: {path_or_index}")
    return cap


def _read_frame_sync(cap: cv2.VideoCapture):
    """Blocking frame read; returns (success, frame)."""
    return cap.read()


async def capture_frames(
    source: str, source_url: Optional[str] = None
) -> AsyncGenerator[bytes, None]:
    """
    Async generator yielding JPEG bytes for each captured frame.

    source: "demo" | "rtsp" | "webcam"
    source_url: required for "rtsp", ignored for others
    """
    loop = asyncio.get_event_loop()
    target_fps = 25.0
    frame_interval = 1.0 / target_fps

    if source == "demo":
        path = settings.DEMO_VIDEO_PATH
        try:
            cap = await loop.run_in_executor(None, _open_capture, path)
        except FrameCaptureError:
            raise FrameCaptureError(
                f"Demo video not found at {path}. "
                "Mount a video to DEMO_VIDEO_PATH."
            )

        try:
            while True:
                t0 = loop.time()
                ret, frame = await loop.run_in_executor(None, _read_frame_sync, cap)
                if not ret:
                    # Loop video: seek back to start
                    await loop.run_in_executor(
                        None, cap.set, cv2.CAP_PROP_POS_FRAMES, 0
                    )
                    continue
                jpeg = await loop.run_in_executor(None, _frame_to_jpeg, frame)
                yield jpeg
                elapsed = loop.time() - t0
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        finally:
            await loop.run_in_executor(None, cap.release)

    elif source == "rtsp":
        if not source_url:
            raise FrameCaptureError("source_url required for RTSP source")
        try:
            cap = await loop.run_in_executor(None, _open_capture, source_url)
        except FrameCaptureError:
            raise

        try:
            while True:
                t0 = loop.time()
                ret, frame = await loop.run_in_executor(None, _read_frame_sync, cap)
                if not ret:
                    raise FrameCaptureError("RTSP stream ended or disconnected")
                jpeg = await loop.run_in_executor(None, _frame_to_jpeg, frame)
                yield jpeg
                elapsed = loop.time() - t0
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        finally:
            await loop.run_in_executor(None, cap.release)

    elif source == "webcam":
        try:
            cap = await loop.run_in_executor(None, _open_capture, 0)
        except FrameCaptureError:
            raise

        try:
            while True:
                t0 = loop.time()
                ret, frame = await loop.run_in_executor(None, _read_frame_sync, cap)
                if not ret:
                    raise FrameCaptureError("Webcam read failed")
                jpeg = await loop.run_in_executor(None, _frame_to_jpeg, frame)
                yield jpeg
                elapsed = loop.time() - t0
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        finally:
            await loop.run_in_executor(None, cap.release)

    else:
        raise FrameCaptureError(f"Unknown source type: {source}")
