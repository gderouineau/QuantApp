import random
import time
from django.conf import settings

class RateLimiter:
    def __init__(self):
        cfg = getattr(settings, "MARKET_RATE_LIMIT", {})
        self.base_sleep = float(cfg.get("base_sleep", 0.2))
        self.jitter = float(cfg.get("jitter", 0.15))
        self._last = 0.0

    def wait(self):
        delay = self.base_sleep + random.random() * self.jitter
        now = time.time()
        elapsed = now - self._last
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last = time.time()

def backoff_sleep(attempt: int):
    time.sleep(0.5 * (2 ** max(0, attempt - 1)))
