from langsmith import traceable

from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.reranker.component import RerankerComponent
from seer.automation.autofix.components.reranker.models import RerankerRequest
from seer.automation.autofix.components.retriever import (
    RetrieverComponent,
    RetrieverOutput,
    RetrieverRequest,
)
from seer.automation.component import BaseComponent


class RetrieverWithRerankerComponent(BaseComponent[RetrieverRequest, RetrieverOutput]):
    context: AutofixContext

    @traceable(name="Retriever With Reranker", run_type="chain", tags=["retriever-reranker:v1"])
    def invoke(self, request: RetrieverRequest) -> RetrieverOutput | None:
        retriever = RetrieverComponent(self.context)

        retriever_output = retriever.invoke(
            request.model_copy(update=dict(include_short_hash_as_id=True))
        )

        if retriever_output is None:
            return None

        reranker = RerankerComponent(self.context)

        reranker_output = reranker.invoke(
            RerankerRequest(query=request.text, chunks=retriever_output.chunks)
        )

        return RetrieverOutput(chunks=reranker_output.chunks)
