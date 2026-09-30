"""Adjust case admission without interrupting active cases."""

import asyncio
import contextlib
from pathlib import Path


class LiveConcurrency:
    def __init__(self, limit: int, path: str, poll_seconds: float = 2):
        if limit < 0:
            raise ValueError("concurrency must be nonnegative")
        self.limit = limit
        self.active = 0
        self.path = Path(path)
        self.poll_seconds = poll_seconds
        self._changed = asyncio.Event()
        self._last_error = None

    def refresh(self):
        try:
            value = self.path.read_text(encoding="utf-8").strip()
            if not value.isascii() or not value.isdecimal():
                raise ValueError("expected a nonnegative integer")
            limit = int(value)
        except (OSError, ValueError) as exc:
            error = str(exc)
            if error != self._last_error:
                print(f"[concurrency] keeping limit={self.limit}: {error}", flush=True)
                self._last_error = error
            return
        self._last_error = None
        if limit != self.limit:
            self.limit = limit
            self._changed.set()
            print(f"[concurrency] limit={limit} active={self.active}", flush=True)

    async def __aenter__(self):
        while self.active >= self.limit:
            self._changed.clear()
            await self._changed.wait()
        self.active += 1
        return self

    async def __aexit__(self, *exc):
        self.active -= 1
        self._changed.set()
        if self.limit == 0 and self.active == 0:
            print("[concurrency] paused; active=0; all active cases finished", flush=True)

    async def _watch(self):
        while True:
            await asyncio.sleep(self.poll_seconds)
            self.refresh()

    @contextlib.asynccontextmanager
    async def monitor(self):
        self.refresh()
        print(
            f"[concurrency] watching {self.path}; limit={self.limit}; "
            "0 pauses new cases and lets active cases finish",
            flush=True,
        )
        task = asyncio.create_task(self._watch())
        try:
            yield self
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
