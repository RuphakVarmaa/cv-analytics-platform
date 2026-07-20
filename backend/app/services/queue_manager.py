import asyncio
import time
from collections import deque
from typing import Optional


class QueueManager:
    """
    Async frame queue with backpressure: if the queue is full when a frame
    arrives, the oldest frame is dropped to make room.
    """

    def __init__(self, maxsize: int = 30):
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=maxsize)
        self._maxsize = maxsize
        self._drop_counter: int = 0
        self._total_puts: int = 0
        # Rolling put timestamps for throughput calculation (last 60s)
        self._put_times: deque = deque(maxlen=10_000)

    async def put_frame(self, frame: bytes) -> bool:
        """
        Enqueue a frame.
        If the queue is full: drop the oldest item, then enqueue.
        Returns True if a frame was dropped, False otherwise.
        """
        dropped = False
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._drop_counter += 1
            dropped = True
        await self._queue.put(frame)
        self._total_puts += 1
        self._put_times.append(time.monotonic())
        return dropped

    async def get_frame(self) -> bytes:
        """Dequeue the next frame, blocking until one is available."""
        return await self._queue.get()

    async def drain(self, timeout: float = 2.0) -> None:
        """
        Drain remaining items from the queue within a timeout.
        Used during shutdown to flush pending frames.
        """
        deadline = time.monotonic() + timeout
        while not self._queue.empty():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(self._queue.get(), timeout=min(0.1, remaining))
            except asyncio.TimeoutError:
                break

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def drop_rate(self) -> float:
        """Fraction of put() calls that resulted in a drop (0.0–1.0)."""
        if self._total_puts == 0:
            return 0.0
        return self._drop_counter / self._total_puts

    @property
    def throughput(self) -> float:
        """Rolling put throughput in frames/sec over the last 10 seconds."""
        now = time.monotonic()
        cutoff = now - 10.0
        recent = [t for t in self._put_times if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        window = recent[-1] - recent[0]
        if window <= 0:
            return 0.0
        return (len(recent) - 1) / window

    def reset_counters(self) -> None:
        self._drop_counter = 0
        self._total_puts = 0
        self._put_times.clear()
