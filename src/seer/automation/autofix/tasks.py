import logging
import os
from typing import Any

import sentry_sdk

from celery_app.app import app as celery_app
from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixRequest
from seer.automation.models import InitializationError
from seer.rpc import SentryRpcClient

logger = logging.getLogger("autofix")


@celery_app.task(time_limit=60 * 60)  # 1 hour task timeout
def run_autofix(data: dict[str, Any]) -> None:
    base_url = os.environ.get("SENTRY_BASE_URL")
    if not base_url:
        raise RuntimeError("SENTRY_BASE_URL must be set")

    client = SentryRpcClient(base_url)

    request = AutofixRequest(**data)
    event_manager = AutofixEventManager(client, request.issue.id)
    try:
        with sentry_sdk.start_span(
            op="seer.automation.autofix",
            description="Run autofix on an issue within celery task",
        ):
            autofix = Autofix(request, event_manager)
            autofix.run()
    except InitializationError as e:
        event_manager.mark_running_steps_errored()
        event_manager.send_autofix_complete(None)
        sentry_sdk.capture_exception(e)
