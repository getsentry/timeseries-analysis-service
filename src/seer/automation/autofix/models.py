import datetime
import enum
from typing import Annotated, Any, Literal, Optional

import sentry_sdk
from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from pydantic.alias_generators import to_camel, to_snake
from typing_extensions import NotRequired, TypedDict

from seer import generator
from seer.automation.agent.models import Usage
from seer.automation.models import FilePatch
from seer.generator import Examples


class StacktraceFrame(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=lambda k: AliasChoices(to_camel(k), to_snake(k)),
            serialization_alias=to_camel,
        )
    )

    function: Annotated[str, Examples(generator.ascii_words)]
    filename: Annotated[str, Examples(generator.file_names)]
    abs_path: Annotated[str, Examples(generator.file_paths)]
    line_no: Optional[int]
    col_no: Optional[int]
    context: list[tuple[int, str]]
    repo_name: Optional[str] = None
    repo_id: Optional[int] = None
    in_app: bool = False


class Stacktrace(BaseModel):
    frames: list[StacktraceFrame]

    def to_str(self, max_frames: int = 16):
        stack_str = ""
        for frame in reversed(self.frames[-max_frames:]):
            col_no_str = f", column {frame.col_no}" if frame.col_no is not None else ""
            repo_str = f" in repo {frame.repo_name}" if frame.repo_name else ""
            line_no_str = (
                f"[Line {frame.line_no}{col_no_str}]"
                if frame.line_no is not None
                else "[Line: Unknown]"
            )
            stack_str += f" {frame.function} in file {frame.filename}{repo_str} {line_no_str} ({'In app' if frame.in_app else 'Not in app'})\n"
            for ctx in frame.context:
                is_suspect_line = ctx[0] == frame.line_no
                stack_str += f"{ctx[1]}{'  <-- SUSPECT LINE' if is_suspect_line else ''}\n"
            stack_str += "------\n"
        return stack_str


class SentryFrame(TypedDict):
    absPath: Optional[str]
    colNo: Optional[int]
    context: list[tuple[int, str]]
    filename: NotRequired[Optional[str]]
    function: NotRequired[Optional[str]]
    inApp: NotRequired[bool]
    instructionAddr: NotRequired[Optional[str]]
    lineNo: NotRequired[Optional[int]]
    module: NotRequired[Optional[str]]
    package: NotRequired[Optional[str]]
    platform: NotRequired[Optional[str]]
    rawFunction: NotRequired[Optional[str]]
    symbol: NotRequired[Optional[str]]
    symbolAddr: NotRequired[Optional[str]]
    trust: NotRequired[Optional[Any]]
    vars: NotRequired[Optional[dict[str, Any]]]
    addrMode: NotRequired[Optional[str]]
    isPrefix: NotRequired[bool]
    isSentinel: NotRequired[bool]
    lock: NotRequired[Optional[Any]]
    map: NotRequired[Optional[str]]
    mapUrl: NotRequired[Optional[str]]
    minGroupingLevel: NotRequired[int]
    origAbsPath: NotRequired[Optional[str]]
    sourceLink: NotRequired[Optional[str]]
    symbolicatorStatus: NotRequired[Optional[Any]]


class SentryStacktrace(TypedDict):
    frames: list[SentryFrame]


class SentryException(BaseModel):
    type: str
    value: str
    stacktrace: Stacktrace

    @field_validator("stacktrace", mode="before")
    @classmethod
    def validate_stacktrace(cls, sentry_stacktrace: SentryStacktrace):
        return Stacktrace.model_validate(sentry_stacktrace)


class SentryExceptionEventData(TypedDict):
    values: list[SentryException]


class SentryExceptionEntry(BaseModel):
    type: Literal["exception"]
    data: SentryExceptionEventData


class SentryEvent(BaseModel):
    title: str
    entries: list[dict]

    # Below fields are not present in the SentryEvent model, but are added for convenience
    exceptions: list[SentryException] = Field(default_factory=list, exclude=True)

    @classmethod
    def from_error_event(cls, error_event: dict):
        exceptions: list[SentryException] = []
        for entry in error_event.get("entries", []):
            if entry.get("type") == "exception":
                for exception in entry.get("data", {}).get("values", []):
                    try:
                        exceptions.append(SentryException.model_validate(exception))
                    except ValidationError:
                        sentry_sdk.capture_exception()
                        continue

        if len(exceptions) == 0:
            raise ValueError("No exceptions found in the event.")

        return cls(**error_event, exceptions=exceptions)


class IssueDetails(BaseModel):
    id: Annotated[int, Examples(generator.unsigned_ints)]
    title: Annotated[str, Examples(generator.ascii_words)]
    short_id: Optional[str] = None
    events: list[SentryEvent]

    @field_validator("events", mode="before")
    @classmethod
    def validate_events(cls, events: list[dict]):
        return [SentryEvent.from_error_event(event) for event in events]


class RepoDefinition(BaseModel):
    provider: Annotated[str, Examples(("github", "integrations:github"))]
    owner: str
    name: str

    @property
    def full_name(self):
        return f"{self.owner}/{self.name}"

    @field_validator("provider", mode="after")
    @classmethod
    def validate_provider(cls, provider: str):
        cleaned_provider = provider
        if provider.startswith("integrations:"):
            cleaned_provider = provider.split(":")[1]

        if cleaned_provider != "github":
            raise ValueError(f"Provider {cleaned_provider} is not supported.")

        return cleaned_provider

    def __hash__(self):
        return hash((self.provider, self.owner, self.name))


class FileChangeError(Exception):
    pass


class ProgressType(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    NEED_MORE_INFORMATION = "NEED_MORE_INFORMATION"
    USER_RESPONSE = "USER_RESPONSE"


class ProgressItem(BaseModel):
    timestamp: str
    message: str
    type: ProgressType
    data: Any = None


class AutofixStatus(enum.Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    CANCELLED = "CANCELLED"

    @classmethod
    def terminal(cls) -> "frozenset[AutofixStatus]":
        return frozenset((cls.COMPLETED, cls.ERROR, cls.CANCELLED))


class ProblemDiscoveryResult(BaseModel):
    status: Literal["CONTINUE", "CANCELLED"]
    description: str
    reasoning: str


class AutofixUserDetails(BaseModel):
    id: Annotated[int, Examples(generator.unsigned_ints)]
    display_name: str


class AutofixRequest(BaseModel):
    organization_id: Annotated[int, Examples(generator.unsigned_ints)]
    project_id: Annotated[int, Examples(generator.unsigned_ints)]
    repos: list[RepoDefinition]
    issue: IssueDetails
    invoking_user: Optional[AutofixUserDetails] = None

    base_commit_sha: Optional[Annotated[str, Examples(generator.shas)]] = None
    additional_context: Optional[str] = None
    timeout_secs: Optional[Annotated[int, Examples((60 * 5,))]] = None
    last_updated: Optional[
        Annotated[datetime.datetime, Examples(datetime.datetime.now() for _ in generator.gen)]
    ] = None

    @property
    def process_request_name(self) -> str:
        return f"autofix:{self.organization_id}:{self.issue.id}"

    @property
    def has_timed_out(self, now: datetime.datetime | None = None) -> bool:
        if self.timeout_secs and self.last_updated:
            if now is None:
                now = datetime.datetime.now()
            return self.last_updated + datetime.timedelta(seconds=self.timeout_secs) < now
        return False

    @field_validator("repos", mode="after")
    @classmethod
    def validate_repo_duplicates(cls, repos):
        if isinstance(repos, list):
            # Check for duplicates by comparing lengths after converting to a set
            if len(set(repos)) != len(repos):
                raise ValueError("Duplicate repos detected in the request.")
            return repos

        raise ValueError("Not a list of repos.")


class AutofixOutput(BaseModel):
    title: str
    description: str
    pr_url: str
    pr_number: int
    repo_name: str
    diff: Optional[list[FilePatch]] = []
    usage: Usage


class AutofixEndpointResponse(BaseModel):
    started: bool


class PullRequestResult(BaseModel):
    pr_number: int
    pr_url: str
    repo: RepoDefinition
    diff: list[FilePatch]


class Step(BaseModel):
    id: str
    title: str

    status: AutofixStatus = AutofixStatus.PENDING

    index: int = -1
    progress: list["ProgressItem | Step"] = Field(default_factory=list)
    completedMessage: Optional[str] = None

    def find_child(self, *, id: str) -> "Step | None":
        for step in self.progress:
            if isinstance(step, Step) and step.id == id:
                return step
        return None

    def find_or_add_child(self, base_step: "Step") -> "Step":
        existing = self.find_child(id=base_step.id)
        if existing:
            return existing

        base_step = base_step.model_copy()
        base_step.index = len(self.progress)
        self.progress.append(base_step)
        return base_step


class AutofixGroupState(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    status: AutofixStatus = AutofixStatus.PENDING
    fix: AutofixOutput | None = None
    completedAt: datetime.datetime | None = None
    usage: Usage = Field(default_factory=Usage)


class AutofixCompleteArgs(BaseModel):
    issue_id: int
    status: AutofixStatus
    steps: list[Step]
    fix: AutofixOutput | None


class AutofixStepUpdateArgs(BaseModel):
    issue_id: int
    status: AutofixStatus
    steps: list[Step]


class AutofixContinuation(AutofixGroupState):
    request: AutofixRequest

    def find_step(self, *, id: str) -> Step | None:
        for step in self.steps:
            if step.id == id:
                return step
        return None

    def find_or_add(self, base_step: Step) -> Step:
        existing = self.find_step(id=base_step.id)
        if existing:
            return existing

        base_step = base_step.model_copy()
        base_step.index = len(self.steps)
        self.steps.append(base_step)
        return base_step

    def mark_all_steps_completed(self):
        for step in self.steps:
            step.status = AutofixStatus.COMPLETED

    def mark_running_steps_errored(self):
        for step in self.steps:
            if step.status == AutofixStatus.PROCESSING:
                step.status = AutofixStatus.ERROR
                for substep in step.progress:
                    if isinstance(substep, Step):
                        if substep.status == AutofixStatus.PROCESSING:
                            substep.status = AutofixStatus.ERROR
                        if substep.status == AutofixStatus.PENDING:
                            substep.status = AutofixStatus.CANCELLED
