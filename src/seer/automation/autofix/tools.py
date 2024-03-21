import json
import logging
import textwrap

from langsmith import traceable

from seer.automation.agent.client import GptClient
from seer.automation.agent.models import Message, Usage
from seer.automation.agent.tools import FunctionTool
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.snippet_replacement import (
    SnippetReplacementComponent,
    SnippetReplacementRequest,
)
from seer.automation.autofix.utils import find_original_snippet
from seer.automation.codebase.codebase_index import CodebaseIndex
from seer.automation.models import FileChange
from seer.automation.state import State

logger = logging.getLogger("autofix")


class BaseTools:
    context: AutofixContext

    def __init__(self, context: AutofixContext):
        self.context = context

    @traceable(run_type="tool", name="Codebase Search")
    def codebase_retriever(self, query: str, repo_name: str | None = None):
        chunks = self.context.query(query, repo_name=repo_name)

        content = ""
        for chunk in chunks:
            content += (
                chunk.get_dump_for_llm(
                    self.context.get_codebase(chunk.repo_id).repo_info.external_slug
                )
                + "\n\n"
            )
        return content

    @traceable(run_type="tool", name="Expand Document")
    def expand_document(self, input: str, repo_name: str | None = None):
        _, document = self.context.get_document_and_codebase(input, repo_name=repo_name)

        if document:
            return document.text

        return "<document with the provided path not found>"

    def get_tools(self):
        return [
            FunctionTool(
                name="codebase_search",
                description=textwrap.dedent(
                    """\
                    Search for code snippets.
                    - You can search for code using either a code snippet or the path.
                    - The codebase is large, so you will need to be very specific with your query.
                    - If the path contains relative paths such as ../, you will need to remove them.
                    - If "code" in "file" search does not work, try searching just the code snippet.

                    Example Queries:
                    - Search for a code snippet: "foo"
                    - Search for a file: "sentry/data/issueTypeConfig/index.tsx"
                    - Search for a function: "getIssueTypeConfig("
                    """
                ),
                parameters=[
                    {
                        "name": "query",
                        "type": "string",
                        "description": "The query to search for.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "Optional name of the repository to search in if you know it.",
                    },
                ],
                fn=self.codebase_retriever,
            ),
            FunctionTool(
                name="expand_document",
                fn=self.expand_document,
                description="Given a document path, returns the entire document text.",
                parameters=[
                    {
                        "name": "input",
                        "type": "string",
                        "description": "The document path to expand.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "Optional name of the repository to search in if you know it.",
                    },
                ],
            ),
        ]


class CodeActionTools(BaseTools):
    snippet_matching_threshold = 0.9
    chunk_padding = 16

    def __init__(self, context: AutofixContext):
        super().__init__(context)

    @traceable(run_type="tool", name="Store File Change")
    def store_file_change(self, codebase: CodebaseIndex, file_change: FileChange):
        """
        Stores a file change to a codebase index.
        This function exists mainly to be traceable in Langsmith.
        """
        codebase.store_file_change(file_change)

    @traceable(run_type="tool", name="Replace Snippet")
    def replace_snippet_with(
        self,
        file_path: str,
        repo_name: str,
        reference_snippet: str,
        replacement_snippet: str,
        commit_message: str,
    ):
        """
        Replaces a snippet with the provided replacement.
        """
        logger.debug(
            f"[CodeActionTools.replace_snippet_with] Replacing snippet\n```\n{reference_snippet}\n```\n with \n```\n{replacement_snippet}\n```\nin {file_path} in repo {repo_name}"
        )

        codebase, document = self.context.get_document_and_codebase(file_path, repo_name=repo_name)

        if not document or not codebase:
            raise FileNotFoundError("File not found or it was deleted in a previous action.")

        result = find_original_snippet(
            reference_snippet, document.text, threshold=self.snippet_matching_threshold
        )

        if not result:
            raise Exception("Reference snippet not found. Try again with an exact match.")

        original_snippet, snippet_start_line, snippet_end_line = result

        lines = document.text.splitlines()
        chunk_lines = lines[
            max(0, snippet_start_line - self.chunk_padding) : min(
                len(lines), snippet_end_line + self.chunk_padding
            )
        ]
        chunk = "\n".join(chunk_lines).strip("\n")

        if not original_snippet:
            raise Exception("Reference snippet not found. Try again with an exact match.")

        output = SnippetReplacementComponent(self.context).invoke(
            SnippetReplacementRequest(
                reference_snippet=original_snippet,
                replacement_snippet=replacement_snippet,
                chunk=chunk,
                commit_message=commit_message,
            )
        )

        if not output:
            raise Exception("Snippet replacement failed.")

        # Add a trailing newline in the reference snippet, this is because we stripped all newlines from the chunk originally, we should add the trailing one back in.
        reference_snippet = chunk + "\n"
        new_snippet = output.snippet
        # Add a trailing snippet to the new snippet to match the reference snippet if there isn't already one.
        if not new_snippet.endswith("\n"):
            new_snippet += "\n"

        self.store_file_change(
            codebase,
            FileChange(
                change_type="edit",
                path=file_path,
                reference_snippet=reference_snippet,
                new_snippet=new_snippet,
                description=commit_message,
            ),
        )

        return f"success: Resulting code after replacement:\n```\n{output.snippet}\n```\n"

    @traceable(run_type="tool", name="Delete Snippet")
    def delete_snippet(self, file_path: str, repo_name: str, snippet: str, commit_message: str):
        """
        Deletes a snippet.
        """
        logger.debug(
            f"[CodeActionTools.delete_snippet] Deleting snippet {snippet} in {file_path} in repo {repo_name}"
        )

        codebase, document = self.context.get_document_and_codebase(file_path)

        if not (document and codebase):
            raise FileNotFoundError("File not found or it was deleted in a previous action.")

        original_snippet: str | None = None
        if snippet in document.text:
            original_snippet = snippet
        else:
            result = find_original_snippet(
                snippet, document.text, threshold=self.snippet_matching_threshold
            )

            if result:
                original_snippet, snippet_start_line, snippet_end_line = result

        logger.debug("Exact snippet:")
        logger.debug(f'"{snippet}"')

        if not original_snippet:
            raise Exception("Reference snippet not found. Try again with an exact match.")
        file_change = FileChange(
            change_type="delete",
            path=file_path,
            description=commit_message,
            reference_snippet=original_snippet,
            new_snippet="",
        )
        self.store_file_change(
            codebase,
            file_change,
        )

        return f"success; New file contents for `{file_path}`: 

```
{file_change.apply(document.text)}
```"

    def modify_existing_file(self, file_path: str, repo_name: str, snippet: str, commit_message: str):
        """
        Modifies an existing file by either appending the new snippet or modifying an existing part of the file.
        """
        logger.debug(
            f"[CodeActionTools.modify_existing_file] Modifying or appending to file {file_path} with snippet {snippet} in {repo_name}"
        )

        codebase, document = self.context.get_document_and_codebase(file_path, repo_name=repo_name)

        if not document or not codebase:
            raise FileNotFoundError("File not found or it was deleted in a previous action.")

        if snippet in document.text:
            logger.info("Snippet already exists in the file.")
            return "Snippet already exists. No changes made."

        new_content = f"{document.text}\n{snippet}"
        self.store_file_change(
            codebase,
            FileChange(
                change_type="edit",
                path=file_path,
                new_snippet=new_content,
                description=commit_message,
            ),
        )

        return "success: Snippet added to the file."
    #         reference_snippet, file_contents, threshold=self._snippet_matching_threshold
    #     )

    #     logger.debug("Exact reference snippet:")
    #     logger.debug(f'"{reference_snippet}"')

    #     if not original_snippet:
    #         raise Exception("Reference snippet not found. Try again with an exact match.")

    #     new_contents = file_contents.replace(original_snippet, original_snippet + "\n" + snippet)

    #     self.context._update_document(file_path, new_contents)

    #     original_contents = file_contents
    #     if file_path in self.file_changes:
    #         original_contents = self.file_changes[file_path].original_contents

    #     self.file_changes[file_path] = FileChange(
    #         change_type="edit",
    #         path=file_path,
    #         contents=new_contents,
    #         description=commit_message,
    #         original_contents=original_contents,
    #     )

    #     return f"success; New file contents for `{file_path}`: \n\n```\n{new_contents}\n```"

    @traceable(run_type="tool", name="Create File")
    def create_file(self, file_path: str, repo_name: str, snippet: str, commit_message: str):
        """
        Creates a file with the provided snippet.
        """
        logger.debug(
            f"[CodeActionTools.create_file] Creating file {file_path} with snippet {snippet} in {repo_name}"
        )

        codebase, document = self.context.get_document_and_codebase(file_path, repo_name=repo_name)

        if not codebase:
            raise FileNotFoundError(f"Repository `{repo_name}` not found.")
        if document:
            raise FileExistsError(f"File `{file_path}` already exists.")

        self.store_file_change(
            codebase,
            FileChange(
                change_type="create",
                path=file_path,
                new_snippet=snippet,
                description=commit_message,
            ),
        )

        return "success"

    @traceable(run_type="tool", name="Delete File")
    def delete_file(self, file_path: str, repo_name: str, commit_message: str):
        """
        Deletes a file.
        """
        logger.debug(f"[CodeActionTools.delete_file] Deleting file {file_path} in {repo_name}")

        codebase, document = self.context.get_document_and_codebase(file_path, repo_name=repo_name)

        if not document or not codebase:
            raise FileNotFoundError(f"File `{file_path}` not found.")

        self.store_file_change(
            codebase,
            FileChange(change_type="delete", path=file_path, description=commit_message),
        )

        return "success"

    def get_retrospection_tools(self):
        return super().get_tools()

    def get_tools(self):
        return super().get_tools() + [
            FunctionTool(
                name="replace_snippet_with",
                description=textwrap.dedent(
                    """\
                    Use this as the primary tool to write code changes to a file.

                    Replaces a snippet in a file with the provided replacement.
                    - The snippet must be an exact match.
                    - The replacement can be any string.
                    - The reference snippet must be an entire line, not just a substring of a line. It should also include the indentation and spacing.
                    - Indentation and spacing must be included in the replacement snippet."""
                ),
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to modify.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "reference_snippet",
                        "type": "string",
                        "description": "The snippet to replace.",
                    },
                    {
                        "name": "replacement_snippet",
                        "type": "string",
                        "description": "The replacement for the snippet.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.replace_snippet_with,
            ),
            FunctionTool(
                name="delete_snippet",
                description=textwrap.dedent(
                    """\
                    Deletes a snippet in a file.
                    - The snippet must be an exact match."""
                ),
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to modify.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "snippet",
                        "type": "string",
                        "description": "The snippet to delete.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.delete_snippet,
            ),
            #             FunctionTool(
            #                 name="insert_snippet",
            #                 description="""Inserts a snippet on a new line directly after reference snippet in a file.
            # - The reference snippet must be an exact match.
            # - Indentation and spacing must be included in the snippet to insert.""",
            #                 parameters=[
            #                     {
            #                         "name": "file_path",
            #                         "type": "string",
            #                         "description": "The file path to modify.",
            #                     },
            #                     {
            #                         "name": "reference_snippet",
            #                         "type": "string",
            #                         "description": "The reference snippet to insert after.",
            #                     },
            #                     {
            #                         "name": "snippet",
            #                         "type": "string",
            #                         "description": "The snippet to insert. This snippet will be on a new line after the reference snippet.",
            #                     },
            #                     {
            #                         "name": "commit_message",
            #                         "type": "string",
            #                         "description": "The commit message to use.",
            #                     },
            #                 ],
            #                 fn=self.insert_snippet,
            #             ),
            FunctionTool(
                name="create_file",
                description="""Creates a file with the provided snippet.""",
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to create.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "snippet",
                        "type": "string",
                        "description": "The snippet to insert.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.create_file,
            ),
            FunctionTool(
                name="delete_file",
                description="Deletes a file.",
                parameters=[
                    {
                        "name": "file_path",
                        "type": "string",
                        "description": "The file path to delete.",
                    },
                    {
                        "name": "repo_name",
                        "type": "string",
                        "description": "The name of the repository to modify.",
                    },
                    {
                        "name": "commit_message",
                        "type": "string",
                        "description": "The commit message to use.",
                    },
                ],
                fn=self.delete_file,
            ),
        ]
