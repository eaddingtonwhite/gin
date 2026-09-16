"""How a runner reaches the service: in-process, or over HTTP. Same calls, same types.

`--service local` exists so the whole loop can be exercised on a laptop with nothing deployed, and
so that what is proven that way is what gets deployed: both clients hand the same JSON to the same
dispatcher. The HTTP client is standard-library `urllib`, because a thin runner with a dependency is
a runner that needs an install step before it can run one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from wire import (ChangeSource, OrderResult, OrdersResponse, RepoFacts, RunRequest, RunTicket, SourceRequest,
                  UploadRequest, UploadTargets, VerdictReport, from_json, to_json)


class ServiceError(RuntimeError):
    """The service refused or could not answer. The message carries its status and body."""


class ServiceClient(Protocol):
    def select(self, facts: RepoFacts, limit: int) -> SourceRequest: ...
    def orders(self, facts: RepoFacts, sources: list[ChangeSource], limit: int) -> OrdersResponse: ...
    def verdicts(self, results: list[OrderResult]) -> VerdictReport: ...
    def uploads(self, request: UploadRequest) -> UploadTargets: ...
    def runs(self, request: RunRequest) -> RunTicket: ...


class LocalService:
    """The service in this process, reached through its dispatcher — not by calling its functions
    directly, so the boundary is exercised even when nothing crosses a network."""

    def __init__(self) -> None:
        from service.app import MemoryStore, Service  # noqa: PLC0415 - only a local runner imports the service

        self._service = Service(MemoryStore())

    def _call(self, path: str, payload: dict[str, Any]) -> Any:
        from service.app import handle  # noqa: PLC0415

        status, body = handle(self._service, "POST", path, {}, json.dumps(payload).encode(), token=None)
        if status != 200:
            raise ServiceError(f"{path} -> {status}: {body.get('error', body)}")
        return body

    def select(self, facts: RepoFacts, limit: int) -> SourceRequest:
        return from_json(SourceRequest, self._call("/v1/select", {"facts": to_json(facts), "limit": limit}))

    def orders(self, facts: RepoFacts, sources: list[ChangeSource], limit: int) -> OrdersResponse:
        payload = {"facts": to_json(facts), "sources": to_json(sources), "limit": limit}
        return from_json(OrdersResponse, self._call("/v1/orders", payload))

    def verdicts(self, results: list[OrderResult]) -> VerdictReport:
        return from_json(VerdictReport, self._call("/v1/verdicts", {"results": to_json(results)}))

    def uploads(self, request: UploadRequest) -> UploadTargets:
        return from_json(UploadTargets, self._call("/v1/uploads", to_json(request)))

    def runs(self, request: RunRequest) -> RunTicket:
        return from_json(RunTicket, self._call("/v1/runs", to_json(request)))


class HttpService:
    """The hosted service."""

    def __init__(self, base_url: str, token: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds

    def _call(self, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"content-type": "application/json", "authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as failure:
            detail = failure.read().decode(errors="replace")[:400]
            raise ServiceError(f"{path} -> {failure.code}: {detail}") from failure
        except urllib.error.URLError as failure:
            raise ServiceError(f"{path}: {failure.reason}") from failure

    def select(self, facts: RepoFacts, limit: int) -> SourceRequest:
        return from_json(SourceRequest, self._call("/v1/select", {"facts": to_json(facts), "limit": limit}))

    def orders(self, facts: RepoFacts, sources: list[ChangeSource], limit: int) -> OrdersResponse:
        payload = {"facts": to_json(facts), "sources": to_json(sources), "limit": limit}
        return from_json(OrdersResponse, self._call("/v1/orders", payload))

    def verdicts(self, results: list[OrderResult]) -> VerdictReport:
        return from_json(VerdictReport, self._call("/v1/verdicts", {"results": to_json(results)}))

    def uploads(self, request: UploadRequest) -> UploadTargets:
        return from_json(UploadTargets, self._call("/v1/uploads", to_json(request)))

    def runs(self, request: RunRequest) -> RunTicket:
        return from_json(RunTicket, self._call("/v1/runs", to_json(request)))


def client_for(service: str, token: str | None) -> ServiceClient:
    """`local`, or a base URL with a bearer token."""
    if service == "local":
        return LocalService()
    if not token:
        raise ServiceError("a hosted service needs a token: pass --token or set MO_EVAL_TOKEN")
    return HttpService(service, token)
