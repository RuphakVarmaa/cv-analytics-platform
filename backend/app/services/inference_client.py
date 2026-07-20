import asyncio
import time
import uuid
from collections import deque
from typing import List, Optional

import httpx
import numpy as np

from app.config import settings
from app.models import DetectionResult, DetectionBox


class InferenceClientError(Exception):
    """Raised when inference fails after retries."""
    pass


class InferenceClient:
    """
    Async HTTP client that POSTs JPEG frames to the ML service /infer endpoint
    and returns structured DetectionResult objects.

    Maintains a circular latency buffer (last 1000 calls) and exposes p50/p95/p99.
    """

    def __init__(self, base_url: Optional[str] = None):
        self._base_url = base_url or settings.ML_SERVICE_URL
        self._client: Optional[httpx.AsyncClient] = None
        self._latencies: deque = deque(maxlen=1000)

    async def __aenter__(self) -> "InferenceClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def start(self) -> None:
        """Start the client (alternative to context manager)."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        )

    async def stop(self) -> None:
        """Stop the client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def infer(self, frame_bytes: bytes, frame_id: Optional[str] = None) -> DetectionResult:
        """
        Post a JPEG frame to /infer.
        Retries once on 5xx with a 100ms delay.
        Records latency in the internal circular buffer.
        """
        if self._client is None:
            raise InferenceClientError("Client not started. Call start() first.")

        if frame_id is None:
            frame_id = str(uuid.uuid4())

        timestamp_ms = int(time.time() * 1000)
        t0 = time.monotonic()

        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = await self._client.post(
                    "/infer",
                    content=frame_bytes,
                    headers={
                        "Content-Type": "image/jpeg",
                        "X-Frame-ID": frame_id,
                    },
                )
                if response.status_code >= 500:
                    if attempt == 0:
                        await asyncio.sleep(0.1)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                data = response.json()

                elapsed_ms = (time.monotonic() - t0) * 1000
                self._latencies.append(elapsed_ms)

                detections = [
                    DetectionBox(**det) for det in data.get("detections", [])
                ]
                return DetectionResult(
                    frame_id=frame_id,
                    timestamp_ms=data.get("timestamp_ms", timestamp_ms),
                    detections=detections,
                    inference_time_ms=data.get("inference_time_ms", elapsed_ms),
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt == 0 and e.response.status_code >= 500:
                    await asyncio.sleep(0.1)
                    continue
                break
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                if attempt == 0:
                    await asyncio.sleep(0.1)
                    continue
                break

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._latencies.append(elapsed_ms)
        raise InferenceClientError(
            f"Inference failed after retries: {last_error}"
        )

    def p50(self) -> float:
        """50th percentile latency in milliseconds."""
        return self._percentile(50)

    def p95(self) -> float:
        """95th percentile latency in milliseconds."""
        return self._percentile(95)

    def p99(self) -> float:
        """99th percentile latency in milliseconds."""
        return self._percentile(99)

    def _percentile(self, pct: float) -> float:
        if not self._latencies:
            return 0.0
        arr = np.array(list(self._latencies))
        return float(np.percentile(arr, pct))

    @property
    def sample_count(self) -> int:
        return len(self._latencies)
