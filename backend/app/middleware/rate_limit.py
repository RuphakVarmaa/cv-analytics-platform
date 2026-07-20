import time
import asyncio
from collections import defaultdict
from typing import Dict, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse


class TokenBucketRateLimiter(BaseHTTPMiddleware):
    """
    Token bucket rate limiter: 30 requests/sec per client IP.
    Uses a sliding window approximation to avoid thundering herd.
    """

    def __init__(self, app, rate: int = 30, window: float = 1.0):
        super().__init__(app)
        self.rate = rate          # tokens per window
        self.window = window      # window size in seconds
        # (tokens, last_refill_time)
        self._buckets: Dict[str, Tuple[float, float]] = defaultdict(lambda: (float(rate), time.monotonic()))
        self._lock = asyncio.Lock()

    def _get_client_key(self, request: Request) -> str:
        # Use forwarded IP if behind proxy, else direct client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health endpoints to avoid false 429s
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        key = self._get_client_key(request)
        now = time.monotonic()

        async with self._lock:
            tokens, last_refill = self._buckets[key]
            elapsed = now - last_refill
            # Refill proportionally to elapsed time
            tokens = min(float(self.rate), tokens + elapsed * (self.rate / self.window))
            last_refill = now

            if tokens < 1.0:
                self._buckets[key] = (tokens, last_refill)
                retry_after = (1.0 - tokens) / (self.rate / self.window)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": f"{retry_after:.2f}"},
                )

            tokens -= 1.0
            self._buckets[key] = (tokens, last_refill)

        return await call_next(request)
