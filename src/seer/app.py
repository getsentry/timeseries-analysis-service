import datetime
import json
import os
import time
from typing import List

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

from seer.automation.autofix.models import AutofixEndpointResponse, AutofixRequest
from seer.automation.autofix.tasks import run_autofix
from seer.bootup import bootup
from seer.db import ProcessRequest, Session
from seer.grouping.grouping import (
    BulkCreateGroupingRecordsResponse,
    CreateGroupingRecordsRequest,
    GroupingRequest,
    SimilarityResponse,
)
from seer.inference_models import embeddings_model, grouping_lookup
from seer.json_api import json_api, register_json_api_views
from seer.severity.severity_inference import SeverityRequest, SeverityResponse
from seer.trend_detection.trend_detector import BreakpointRequest, BreakpointResponse, find_trends

app = bootup(
    __name__,
    [FlaskIntegration()],
    init_migrations=True,
    eager_load_inference_models=os.environ.get("LAZY_INFERENCE_MODELS") != "1",
)


@json_api("/v0/issues/severity-score")
def severity_endpoint(data: SeverityRequest) -> SeverityResponse:
    if data.trigger_error:
        raise Exception("oh no")
    elif data.trigger_timeout:
        time.sleep(0.5)

    with sentry_sdk.start_span(
        op="seer.severity",
        description="Generate issue severity score",
    ) as span:
        response = embeddings_model().severity_score(data)
        span.set_tag("severity", str(response.severity))
    return response


@json_api("/trends/breakpoint-detector")
def breakpoint_trends_endpoint(data: BreakpointRequest) -> BreakpointResponse:
    txns_data = data.data

    sort_function = data.sort
    allow_midpoint = data.allow_midpoint == "1"
    validate_tail_hours = data.validate_tail_hours

    min_pct_change = data.trend_percentage
    min_change = data.min_change

    with sentry_sdk.start_span(
        op="seer.breakpoint_detection",
        description="Get the breakpoint and t-value for every transaction",
    ) as span:
        trend_percentage_list = find_trends(
            txns_data,
            sort_function,
            allow_midpoint,
            min_pct_change,
            min_change,
            validate_tail_hours,
        )

    trends = BreakpointResponse(data=[x[1] for x in trend_percentage_list])
    app.logger.debug("Trend results: %s", trends)

    return trends


@json_api("/v0/issues/similar-issues")
def similarity_endpoint(data: GroupingRequest) -> SimilarityResponse:
    with sentry_sdk.start_span(op="seer.grouping", description="grouping lookup") as span:
        similar_issues = grouping_lookup().get_nearest_neighbors(data)
    return similar_issues


@json_api("/v0/issues/similar-issues/grouping-record")
def similarity_grouping_record_endpoint(
    data: CreateGroupingRecordsRequest,
) -> BulkCreateGroupingRecordsResponse:
    with sentry_sdk.start_span(
        op="seer.grouping-record", description="grouping record bulk insert"
    ) as span:
        success = grouping_lookup().bulk_create_and_insert_grouping_records(data)
    return success


@json_api("/v0/automation/autofix")
def autofix_endpoint(data: AutofixRequest) -> AutofixEndpointResponse:
    run_autofix.delay(data.model_dump(mode="json"))
    return AutofixEndpointResponse(started=True)


@app.route("/health/live", methods=["GET"])
def health_check():
    return "", 200


@app.route("/health/ready", methods=["GET"])
def ready_check():
    return "", 200


register_json_api_views(app)
