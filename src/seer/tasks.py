import asyncio
import contextlib
import dataclasses
import datetime
import hashlib
import logging
import threading
from asyncio import Future, Task
from queue import Queue
from typing import Any, Callable, Coroutine, Protocol, TypeVar

import sqlalchemy
from dateutil.relativedelta import relativedelta
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sqlalchemy import func, select, text
from sqlalchemy.exc import TransactionTimeoutError
from sqlalchemy.ext.asyncio import async_sessionmaker

from seer.db import AsyncSession, ProcessRequest
from seer.task_factory import AsyncTaskFactory, _async_task_factories

_A = TypeVar("_A")

logger = logging.getLogger("asyncrunner")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class QConsumer(Protocol):
    def get(self, block: bool = True, timeout: int | None = None) -> Any:
        ...


@dataclasses.dataclass(frozen=True, order=True)
class Period:
    period_start: datetime.date
    period: relativedelta

    @property
    def period_end(self) -> datetime.date:
        return self.period_start + self.period - datetime.timedelta(days=1)

    def fit_to(self, date: datetime.date) -> "Period":
        result = self

        while date < result.period_start:
            result = result.prev()

        while date > result.period_end:
            result = result.next()

        return result

    def next(self) -> "Period":
        return Period(period_start=self.period_start + self.period, period=self.period)

    def prev(self) -> "Period":
        return Period(period_start=self.period_start - self.period, period=self.period)


@dataclasses.dataclass
class AsyncApp:
    end_event: asyncio.Event = dataclasses.field(default_factory=lambda: asyncio.Event())
    num_consumers: int = 10
    queue: asyncio.Queue = dataclasses.field(default_factory=lambda: asyncio.Queue())
    task_factories: list[Callable[[], AsyncTaskFactory]] = dataclasses.field(
        default_factory=lambda: _async_task_factories,
    )
    completed_queue: Queue | None = None
    consumer_sleep: int = 5
    fail_fast: bool = False

    async def run_or_end(self, c: Future[_A] | Coroutine[Any, Any, _A]) -> tuple[_A] | None:
        end_task = asyncio.create_task(self.end_event.wait())
        task: Future[_A] | Task[_A]
        if not isinstance(c, Future):
            task = asyncio.create_task(c)
        else:
            task = c
        parts: list[Future | Task] = [end_task, task]
        await asyncio.wait(parts, return_when=asyncio.FIRST_COMPLETED)
        if end_task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return None
        else:
            end_task.cancel()
        if exc := task.exception():
            raise exc
        return (task.result(),)

    async def select_from_db(self) -> None:
        while not self.end_event.is_set():
            async with AsyncSession() as session:
                logger.info("Checking for process requests")
                result = await self.run_or_end(
                    session.run_sync(
                        lambda session: ProcessRequest.acquire_work(
                            10, datetime.datetime.utcnow(), session=session
                        )
                    )
                )
            if result is not None and result[0]:
                for item in result[0]:
                    logger.info(f"Picked up process request, running")
                    await self.run_or_end(self.queue.put(item))
                    logger.info(f"Process request completed successfully")
                    if self.end_event.is_set():
                        break
            else:
                logger.info("Sleeping")
                await self.run_or_end(asyncio.sleep(self.consumer_sleep))

    async def producer_loop(self):
        await asyncio.gather(self.select_from_db())

    async def invoke_task(self, task: AsyncTaskFactory, process_request: ProcessRequest):
        try:
            async with AsyncSession() as session, acquire_x_lock(
                process_request.name, session
            ) as acquired:
                if acquired:
                    await session.execute(text("SET idle_in_transaction_session_timeout = '1h'"))
                    await task.invoke(process_request)
                return acquired
        except TransactionTimeoutError:
            logger.error("Transaction timeout occurred. Failing gracefully.")
            # Additional logic for failing gracefully can be added here

    async def consumer_loop(self):
        while not self.end_event.is_set():
            logger.info("Running consumer loop")
            result = await self.run_or_end(self.queue.get())
            if result is None:
                continue

            item: ProcessRequest = result[0]
            logger.info(f"Consumer loop working on {item.name}.")
            for factory in self.task_factories:
                task = factory()
                accept = task.matches(item)
                if accept:
                    result = await self.run_or_end(self.invoke_task(task, item))

                    if result:
                        if item.id:
                            async with AsyncSession() as session:
                                logger.info("Marking process request completed.")
                                await session.execute(item.mark_completed_stmt())
                                await session.commit()
                        if self.completed_queue:
                            q = self.completed_queue
                            await self.run_or_end(
                                asyncio.get_event_loop().run_in_executor(None, lambda: q.put(item))
                            )

    async def kill_event_task(self, kill_event: threading.Event | None):
        if kill_event is None:
            return

        while not self.end_event.is_set():
            acquired = await asyncio.get_event_loop().run_in_executor(
                None, lambda: kill_event.wait(1)
            )
            if acquired:
                self.end_event.set()
                break

    async def run(self, kill_event: threading.Event | None = None):
        # Force loading of tasks
        from seer.automation.autofix import tasks  # noqa

        kill_task = asyncio.create_task(self.kill_event_task(kill_event))
        producer_task = asyncio.create_task(self.producer_loop())
        consumer_tasks: list[asyncio.Task[Any]] = []
        while not self.end_event.is_set():
            if producer_task.done():
                if self.fail_fast:
                    self.end_event.set()
                else:
                    logger.info("Unexpected producer death, restarting...")
                    producer_task = asyncio.create_task(self.producer_loop())
                continue

            task: asyncio.Task
            for task in [*consumer_tasks]:
                if task.done():
                    if self.fail_fast:
                        self.end_event.set()
                    else:
                        logger.info("Unexpected consumer death, restarting...")
                        consumer_tasks.remove(task)

            while len(consumer_tasks) < self.num_consumers:
                consumer_tasks.append(asyncio.create_task(self.consumer_loop()))
            await self.run_or_end(asyncio.sleep(1))

        all_tasks = [producer_task, *consumer_tasks, kill_task]
        async with asyncio.timeout(10):
            await asyncio.gather(*all_tasks)


@contextlib.asynccontextmanager
async def acquire_x_lock(name: str, session: sqlalchemy.ext.asyncio.AsyncSession):
    m = hashlib.sha256()
    m.update(name.encode("utf8"))
    key = int.from_bytes(m.digest()[:8], byteorder="big", signed=True) & 0xFFFFFFFF

    logger.info("Acquiring exclusive lock for %s", key)
    rows = await session.execute(select(func.pg_try_advisory_xact_lock(key)))
    acquired = next(rows)[0]
    yield acquired


async def async_main():
    from seer.bootup import bootup

    bootup(
        __name__,
        [AsyncioIntegration()],
        init_db=True,
        with_async=True,
        eager_load_inference_models=False,
    )
    app = AsyncApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(async_main())
