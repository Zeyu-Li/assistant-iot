"""The only way adapters get an HTTP client: every request is checked against a host allowlist.

The assistant has no general internet access. A host that is not listed in the
profile's ``network.allowed_hosts`` (or a loopback address) fails before any bytes leave.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import httpx
import httpx2

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class EgressBlocked(RuntimeError):
    pass


class HttpClientFactory:
    def __init__(self, allowed_hosts: Iterable[str]) -> None:
        self._allowed = frozenset(h.lower() for h in allowed_hosts) | LOOPBACK_HOSTS

    def is_allowed(self, host: str) -> bool:
        return host.lower() in self._allowed

    async def _check(self, request: Any) -> None:
        host = request.url.host
        if not self.is_allowed(host):
            raise EgressBlocked(f"blocked request to {host}: not in network.allowed_hosts")

    def create(
        self,
        *,
        timeout: float,
        base_url: str = "",
        headers: Optional[dict[str, str]] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            event_hooks={"request": [self._check]},
        )

    def create_for_mcp(self, *, timeout: float, headers: Optional[dict[str, str]] = None) -> httpx2.AsyncClient:
        """The MCP SDK takes an httpx2 client (Pydantic's httpx fork); same allowlist check."""
        return httpx2.AsyncClient(
            headers=headers,
            timeout=httpx2.Timeout(timeout, read=max(timeout, 300.0)),
            follow_redirects=False,
            event_hooks={"request": [self._check]},
        )
