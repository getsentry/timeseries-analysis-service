import contextlib
import dataclasses
import hashlib
import hmac
import logging
import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable

import aiohttp
import requests
from aiohttp import ContentTypeError
from aiohttp.http_exceptions import HttpProcessingError
from requests import HTTPError

from seer.utils import json_dumps
from seer.utils import SeerJSONEncoder

logger = logging.getLogger(__name__)


class RpcClient(ABC):
    @abstractmethod
    def call(self, method: str, **kwargs) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def acall(self, method: str, **kwargs) -> dict[str, Any] | None:
        pass


@dataclasses.dataclass
class FakeHttpResponse:
    status_code: int
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


RpcClientHandler = Callable[[str, dict[str, Any]], dict[str, Any] | tuple[int, str] | None]


@dataclasses.dataclass
class DummyRpcClient(RpcClient):
    handlers: dict[str, RpcClientHandler] = dataclasses.field(default_factory=dict)
    missed_calls: list[tuple[str, dict[str, Any]]] = dataclasses.field(default_factory=list)
    should_log: bool = False
    dry_run: bool = False

    def call(self, method: str, **kwargs) -> dict[str, Any] | None:
        result = self.handlers.get(method, self._default_call)(method, kwargs)
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        status, msg = result
        raise HTTPError(response=FakeHttpResponse(status_code=status, content=msg.encode("utf-8")))

    def _default_call(
        self, method: str, kwargs: dict[str, Any]
    ) -> dict[str, Any] | tuple[int, str] | None:
        if self.dry_run:
            return None

        self.missed_calls.append((method, kwargs))
        body_dict = {"args": kwargs}
        json_dump = json_dumps(body_dict, separators=(",", ":"))

        if self.should_log:
            print(f"Calling {method} with {json_dump}")
        return 404, "Not Found"

    async def acall(self, method: str, **kwargs) -> dict[str, Any] | None:
        kwargs.pop("session", None)
        result = self.handlers.get(method, self._default_call)(method, kwargs)
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        status, msg = result
        raise HttpProcessingError(code=status, message=msg)


class SentryRpcClient(RpcClient):
    @cached_property
    def shared_secret(self) -> str:
        shared_secret = os.environ.get("RPC_SHARED_SECRET")
        if not shared_secret:
            raise RuntimeError("RPC_SHARED_SECRET must be set")
        return shared_secret

    @cached_property
    def base_url(self) -> str:
        base_url = os.environ.get("SENTRY_BASE_URL")
        if not base_url:
            raise RuntimeError("SENTRY_BASE_URL must be set")
        return base_url

    def _generate_request_signature(self, url_path: str, body: bytes) -> str:
        signature_input = b"%s:%s" % (url_path.encode("utf8"), body)
        signature = hmac.new(
            self.shared_secret.encode("utf-8"), signature_input, hashlib.sha256
        ).hexdigest()
        return f"rpc0:{signature}"

    def call(self, method: str, **kwargs) -> dict[str, Any] | None:
        body_bytes, endpoint, headers = self._prepare_request(method, kwargs)
        response = requests.post(endpoint, headers=headers, data=body_bytes)
        response.raise_for_status()
        if response.headers.get("Content-Type", "") != "application/json":
            logger.warning("No application/json content type")
            return None
        return response.json()

    async def acall(
        self, method: str, session: aiohttp.ClientSession | None = None, **kwargs
    ) -> dict[str, Any] | None:
        body_bytes, endpoint, headers = self._prepare_request(method, kwargs)
        async with contextlib.AsyncExitStack() as stack:
            if session is None:

                session = aiohttp.ClientSession()
            await stack.enter_async_context(session)
            response = await stack.enter_async_context(
                session.post(endpoint, headers=headers, data=body_bytes)
            )
            response.raise_for_status()
            try:
                result = await response.json()
            except ContentTypeError:
                result = None
        return result

    def _prepare_request(self, method, kwargs):
        url_path = f"/api/0/internal/seer-rpc/{method}/"
        endpoint = f"{self.base_url}{url_path}"
        body_dict = {"args": kwargs}
        body = json.dumps(body_dict, cls=SeerJSONEncoder, separators=(",", ":"))
        body_bytes = body.encode("utf-8")
        signature = self._generate_request_signature(url_path, body_bytes)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Rpcsignature {signature}",
        }
        return body_bytes, endpoint, headers
