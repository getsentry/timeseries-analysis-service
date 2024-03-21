import uuid

from sentence_transformers import SentenceTransformer

from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import (
    AutofixContinuation,
    EventDetails,
    RepoDefinition,
    Stacktrace,
)
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.codebase.models import StoredDocumentChunk
from seer.automation.pipeline import PipelineContext
from seer.automation.state import State
from seer.automation.utils import get_embedding_model
from seer.db import DbDocumentChunk, Session


class AutofixContext(PipelineContext):
    state: State[AutofixContinuation]
    codebases: dict[int, CodebaseIndex]
    event_manager: AutofixEventManager

    commit_changes: bool = True

    def __init__(
        self,
        organization_id: int,
        project_id: int,
        repos: list[RepoDefinition],
        event_manager: AutofixEventManager,
        state: State[AutofixContinuation],
        embedding_model: SentenceTransformer | None = None,
    ):
        self.organization_id = organization_id
        self.project_id = project_id
        self.run_id = uuid.uuid4()
        self.codebases = {}

        self.embedding_model = embedding_model or get_embedding_model()

        for repo in repos:
            codebase_index = CodebaseIndex.from_repo_definition(
                organization_id, project_id, repo, self.run_id, embedding_model=self.embedding_model
            )

            if codebase_index:
                self.codebases[codebase_index.repo_info.id] = codebase_index

        self.event_manager = event_manager
        self.state = state

    def has_codebase_index(self, repo: RepoDefinition) -> bool:
        return CodebaseIndex.has_repo_been_indexed(self.organization_id, self.project_id, repo)

    def create_codebase_index(self, repo: RepoDefinition):
        codebase_index = CodebaseIndex.create(
            self.organization_id,
            self.project_id,
            repo,
            self.run_id,
            embedding_model=self.embedding_model,
        )
        self.codebases[codebase_index.repo_info.id] = codebase_index

    def get_codebase(self, repo_id: int) -> CodebaseIndex:
        codebase = self.codebases[repo_id]

        if codebase is None:
            raise ValueError(f"Codebase with id {repo_id} not found")

        return codebase

    def query(
        self, query: str, repo_name: str | None = None, repo_id: int | None = None, top_k: int = 8
    ):
        if repo_name:
            repo_id = next(
                (
                    repo_id
                    for repo_id, codebase in self.codebases.items()
                    if codebase.repo_info.external_slug == repo_name
                ),
                None,
            )

        repo_ids = [repo_id] if repo_id is not None else list(self.codebases.keys())

        embedding = self.embedding_model.encode(query)

        with Session() as session:
            db_chunks = (
                session.query(DbDocumentChunk)
                .filter(
                    DbDocumentChunk.repo_id.in_(repo_ids),
                    (DbDocumentChunk.namespace == str(self.run_id))
                    | (DbDocumentChunk.namespace.is_(None)),
                )
                .order_by(DbDocumentChunk.embedding.cosine_distance(embedding))
                .limit(top_k)
                .all()
            )

            chunks_by_repo_id: dict[int, list[DbDocumentChunk]] = {}
            for db_chunk in db_chunks:
                chunks_by_repo_id.setdefault(db_chunk.repo_id, []).append(db_chunk)

            populated_chunks: list[StoredDocumentChunk] = []
            for _repo_id, db_chunks_for_codebase in chunks_by_repo_id.items():
                codebase = self.get_codebase(_repo_id)
                populated_chunks.extend(codebase._populate_chunks(db_chunks_for_codebase))

            # Re-sort populated_chunks based on their original order in db_chunks
            db_chunk_order = {db_chunk.id: index for index, db_chunk in enumerate(db_chunks)}
            populated_chunks.sort(key=lambda chunk: db_chunk_order.get(chunk.id, float('inf')))

        return populated_chunks

    def get_document_and_codebase(
        self, path: str, repo_name: str | None = None, repo_id: int | None = None
    ):
        if repo_name:
            repo_id = next(
                (
                    repo_id
                    for repo_id, codebase in self.codebases.items()
                    if codebase.repo_info.external_slug == repo_name
                ),
                None,
            )
        if repo_id:
            codebase = self.get_codebase(repo_id)
            return codebase, codebase.get_document(path)

        for codebase in self.codebases.values():
            document = codebase.get_document(path)
            if document:
                return codebase, document

        return None, None

    def diff_contains_stacktrace_files(self, repo_id: int, event_details: EventDetails) -> bool:
        stacktraces = [exception.stacktrace for exception in event_details.exceptions]

        stacktrace_files: set[str] = set()
        for stacktrace in stacktraces:
            for frame in stacktrace.frames:
                stacktrace_files.add(frame.filename)

        codebase = self.get_codebase(repo_id)
        changed_files, removed_files = codebase.repo_client.get_commit_file_diffs(
            codebase.repo_info.sha, codebase.repo_client.get_default_branch_head_sha()
        )

        change_files = set(changed_files + removed_files)

        return bool(change_files.intersection(stacktrace_files))

    def _process_stacktrace_paths(self, stacktrace: Stacktrace):
        """
        Annotate a stacktrace with the correct repo each frame is pointing to and fix the filenames
        """
        for codebase in self.codebases.values():
            codebase.process_stacktrace(stacktrace)

    def process_event_paths(self, event: EventDetails):
        """
        Annotate exceptions with the correct repo each frame is pointing to and fix the filenames
        """
        for exception in event.exceptions:
            self._process_stacktrace_paths(exception.stacktrace)

    def cleanup(self):
        for codebase in self.codebases.values():
            codebase.cleanup()
