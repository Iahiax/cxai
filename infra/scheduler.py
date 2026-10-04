from __future__ import annotations
import asyncio, logging, signal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

log = logging.getLogger("scheduler")


@dataclass
class Job:
    name: str
    every_seconds: int
    fn: Callable[[], Awaitable | None]
    run_on_start: bool = False
    last_run: float = 0.0
    running: bool = False
    errors: int = 0


class AsyncScheduler:
    def __init__(self):
        self.jobs: list[Job] = []
        self._stop = asyncio.Event()

    def add(self, name: str, every_seconds: int, fn,
            run_on_start: bool = False):
        self.jobs.append(Job(name, every_seconds, fn, run_on_start))

    async def _run_job(self, job: Job):
        job.running = True
        try:
            result = job.fn()
            if asyncio.iscoroutine(result):
                await result
            job.last_run = datetime.now(timezone.utc).timestamp()
            job.errors = 0
        except Exception as e:
            job.errors += 1
            log.exception("job %s failed (#%d): %s", job.name, job.errors, e)
        finally:
            job.running = False

    async def _loop(self):
        while not self._stop.is_set():
            now = datetime.now(timezone.utc).timestamp()
            for job in self.jobs:
                if job.running:
                    continue
                due = (now - job.last_run) >= job.every_seconds
                initial = job.run_on_start and job.last_run == 0
                if due or initial:
                    asyncio.create_task(self._run_job(job))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    def stop(self):
        self._stop.set()

    def status(self) -> list[dict]:
        now = datetime.now(timezone.utc).timestamp()
        return [{
            "name": j.name, "running": j.running,
            "last_run_ago": int(now - j.last_run) if j.last_run else None,
            "errors": j.errors, "every": j.every_seconds,
        } for j in self.jobs]

    async def serve(self):
        try:
            await self._loop()
        finally:
            log.info("scheduler stopped")


def install_signal_handlers(sched: AsyncScheduler,
                            loop: asyncio.AbstractEventLoop):
    def _handler():
        log.warning("signal received, stopping scheduler")
        sched.stop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except NotImplementedError:
            pass
