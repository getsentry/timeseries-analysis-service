import asyncio
import dataclasses
import datetime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Annotated, Callable, Self

import pytest
from pydantic import BaseModel
from sqlalchemy import select

from seer.db import ProcessRequest, Session
from seer.generator import (
    Examples,
    ascii_words,
    change_watcher,
    gen,
    generate,
    ints,
    parameterize,
    positive_timedeltas,
    printable_strings,
)
from seer.tasks import AsyncApp, AsyncSession, AsyncTaskFactory
from tests.generators import Future, Now, Past


@dataclasses.dataclass
class ScheduledWork:
    process_request: ProcessRequest
    name: Annotated[str, Examples(s for s in ascii_words if s)]
    payload: dict[str, str]
    now: Now

    def __post_init__(self):
        self.process_request.name = self.name
        self.process_request.scheduled_from = self.now
        self.process_request.scheduled_for = self.now

    def save(self) -> Self:
        with Session() as session:
            session.add(self.process_request)
            session.commit()
        return self

    def reload(self) -> Self:
        with Session() as session:
            self.process_request = session.merge(self.process_request, load=True)
            session.refresh(self.process_request)
        return self


@dataclasses.dataclass
class UpdatedWork:
    scheduled_work: ScheduledWork
    new_payload: Annotated[dict, Examples(({"this-unique-payload": i} for i in ints))]

    @property
    def original_process_request(self):
        return self.scheduled_work.process_request

    @property
    def current_process_request_by_name(self):
        with Session() as session:
            return session.scalar(
                select(ProcessRequest).where(
                    ProcessRequest.name == self.original_process_request.name
                )
            )

    def save(
        self,
        now: datetime.datetime,
        expected_duration: datetime.timedelta = datetime.timedelta(seconds=0),
    ) -> Self:
        self.scheduled_work.save()
        with Session() as session:
            session.execute(
                ProcessRequest.schedule_stmt(
                    self.original_process_request.name,
                    self.new_payload,
                    when=now,
                    expected_duration=expected_duration,
                )
            )
            session.commit()
            return self


@parameterize
def test_schedule_prefers_urgency(updated: UpdatedWork, future: Future, past: Past):
    updated.scheduled_work.save()

    payload_watcher = change_watcher(lambda: updated.current_process_request_by_name.payload)
    scheduled_watcher = change_watcher(
        lambda: (
            updated.current_process_request_by_name.scheduled_for,
            updated.current_process_request_by_name.scheduled_from,
        )
    )

    with payload_watcher as payload_changes, scheduled_watcher as schedule_changes:
        updated.save(future)

    assert not schedule_changes
    assert payload_changes.to_value(updated.new_payload)

    with payload_watcher as payload_changes, scheduled_watcher as schedule_changes:
        updated.new_payload["new_thing"] = 1
        updated.save(past)

    assert schedule_changes
    assert payload_changes.to_value(updated.new_payload)


@parameterize(duration=(d for d in positive_timedeltas if d > datetime.timedelta(minutes=1)))
def test_schedule_preserves_expected_duration(
    updated: UpdatedWork, future: Future, duration: datetime.timedelta
):
    updated.scheduled_work.save()

    with change_watcher(
        lambda: updated.current_process_request_by_name.last_delay()
    ) as delay_changes, change_watcher(
        lambda: updated.current_process_request_by_name.scheduled_for
    ) as for_changes:
        updated.save(future, duration)

    assert not for_changes
    assert delay_changes.to_value(duration)


@parameterize
def test_mark_complete_does_not_erase_concurrent_work(updated: UpdatedWork):
    updated.save(datetime.datetime.now() + datetime.timedelta(seconds=1))
    with Session() as session:
        session.execute(updated.original_process_request.mark_completed_stmt())

    assert updated.current_process_request_by_name is not None

    with Session() as session:
        session.execute(updated.current_process_request_by_name.mark_completed_stmt())
        session.commit()

    assert updated.current_process_request_by_name is None


@parameterize
def test_next_schedule(
    scheduled: tuple[ScheduledWork, ScheduledWork, ScheduledWork, ScheduledWork]
):
    for s in scheduled:
        s.save()

    work_1 = ProcessRequest.acquire_work(2, datetime.datetime.now())
    assert len(work_1) == 2

    work_2 = ProcessRequest.acquire_work(2, datetime.datetime.now())
    assert len(work_2) == 2

    assert len(ProcessRequest.acquire_work(4, datetime.datetime.now())) == 0

    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=1)))
        == 4
    )
    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=1)))
        == 0
    )

    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=2)))
        == 0
    )
    assert (
        len(ProcessRequest.acquire_work(4, datetime.datetime.now() + datetime.timedelta(minutes=3)))
        == 4
    )


@parameterize
def test_next_schedule(scheduled: ScheduledWork):
    scheduled.save()
    proc = scheduled.process_request

    last_delay_watcher = change_watcher(lambda: proc.last_delay())

    with last_delay_watcher as changed:
        proc.scheduled_for = proc.next_schedule(proc.scheduled_from)
    assert changed.to_value(datetime.timedelta(minutes=2))

    with last_delay_watcher as changed:
        proc.scheduled_for = proc.next_schedule(proc.scheduled_from)
    assert changed.to_value(datetime.timedelta(minutes=4))

    with last_delay_watcher as changed:
        proc.scheduled_for = proc.next_schedule(proc.scheduled_from)
    assert changed.to_value(datetime.timedelta(minutes=8))


@parameterize
def test_peek_scheduled(scheduled: ScheduledWork, future: Future):
    peek_watcher = change_watcher(ProcessRequest.peek_next_scheduled)

    with peek_watcher as changes:
        scheduled.save()

    assert changes.from_value(None)

    with peek_watcher as changes:
        work = ProcessRequest.acquire_work(1, future)
        assert work, "did not acquire work"
        scheduled.reload()

    assert changes.to_value(work[0].scheduled_for)
    assert changes.to_value(scheduled.process_request.scheduled_for)


class TestRequest(BaseModel):
    my_special_value: str


class TestAsyncTaskFactory(AsyncTaskFactory):
    acceptible_names: frozenset[str] = frozenset(["job-1", "job-2", "job-3", "job-4"])
    side_effect: Callable[[str], None] = lambda _: None

    def matches(self, process_request: ProcessRequest):
        return process_request.name in self.acceptible_names

    async def invoke(self, process_request: ProcessRequest):
        self.side_effect(TestRequest(**process_request.payload).my_special_value)


@dataclasses.dataclass
class ScheduleAsyncTest:
    acceptable_name: Annotated[str, Examples(TestAsyncTaskFactory.acceptible_names)]
    unacceptable_name: Annotated[
        str,
        Examples((s for s in printable_strings if s not in TestAsyncTaskFactory.acceptible_names)),
    ]
    payload: TestRequest
    side_effect_calls: list[str] = dataclasses.field(default_factory=list)
    end_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    def create_app(self, io_work=multiprocessing.Queue()) -> AsyncApp:
        test_task = TestAsyncTaskFactory()
        test_task.side_effect = lambda v: self.side_effect_calls.append(v)

        return AsyncApp(
            end_event=self.end_event,
            queue=asyncio.Queue(),
            task_factories=[
                lambda: test_task,
            ],
            num_consumers=2,
        )

    async def current_process_request_by_name(self, name: str):
        async with AsyncSession() as session:
            return await session.scalar(select(ProcessRequest).where(ProcessRequest.name == name))

    async def new_side_effect(self):
        self.side_effect_calls = []
        for _ in range(20):
            await asyncio.sleep(0.1)
            if self.side_effect_calls:
                return self.side_effect_calls
        raise ValueError(f"side effect of invoke was not invoked for 2 seconds")


@pytest.mark.asyncio
@parameterize
async def test_next_schedule_async(test: ScheduleAsyncTest, now: Now):
    async with AsyncSession() as session:
        for name in [test.acceptable_name, test.unacceptable_name]:
            await session.execute(
                ProcessRequest.schedule_stmt(name, test.payload.model_dump(), when=now)
            )
        await session.commit()

    process = await test.current_process_request_by_name(test.acceptable_name)
    assert process is not None

    run_task = asyncio.create_task(test.create_app().run())
    processed = await test.new_side_effect()
    test.end_event.set()

    async with asyncio.timeout(10):
        await run_task

    assert processed == [test.payload.my_special_value]
    process = await test.current_process_request_by_name(test.acceptable_name)
    assert process is None

    process = await test.current_process_request_by_name(test.unacceptable_name)
    assert process is not None
