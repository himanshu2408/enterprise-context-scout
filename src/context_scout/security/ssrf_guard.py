"""Connect-time DNS-pinning SSRF defense and origin validation (ECS-032, ECS-041, ECS-122)."""

from __future__ import annotations

import ipaddress
import socket
import ssl
from typing import Any
from urllib.parse import SplitResult, urlsplit

import httpcore
import httpx

METADATA_HOSTNAMES: frozenset[str] = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)
METADATA_IP_LITERALS: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "100.100.100.200",
        "fd00:ec2::254",
    }
)
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class SSRFBlockedError(PermissionError):
    """Raised fail-closed when an outbound URL or resolved peer IP violates SSRF policy."""


def validate_ip_address(
    ip_raw: str | ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    raw_host: str = "",
    allow_private_networks: bool = False,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Validates a resolved IP address against loopback, metadata, link-local, and private nets."""
    ip_obj = ipaddress.ip_address(ip_raw) if isinstance(ip_raw, str) else ip_raw
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped

    host_label = raw_host or str(ip_obj)
    if str(ip_obj) in METADATA_IP_LITERALS:
        raise SSRFBlockedError(
            f"Outbound request blocked to cloud metadata IP {ip_obj} (host={host_label!r})"
        )
    if ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified:
        raise SSRFBlockedError(
            f"Outbound request blocked to unsafe peer IP {ip_obj} (host={host_label!r})"
        )
    if not allow_private_networks:
        in_cgnat = isinstance(ip_obj, ipaddress.IPv4Address) and ip_obj in _CGNAT_NETWORK
        if ip_obj.is_private or ip_obj.is_reserved or in_cgnat:
            raise SSRFBlockedError(
                f"Outbound request blocked to private/reserved IP {ip_obj} (host={host_label!r})"
            )
    return ip_obj


def validate_url_target(
    raw_url: str,
    *,
    allow_private_networks: bool = False,
) -> str:
    """Validates URL scheme, syntax, metadata hosts, and literal peer IPs."""
    cleaned = raw_url.strip()
    if not cleaned:
        raise ValueError("Outbound URL must not be empty.")
    if any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        raise SSRFBlockedError(
            f"Outbound URL contains forbidden control/whitespace chars: {raw_url!r}"
        )

    parsed = urlsplit(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SSRFBlockedError(
            f"Outbound URL must use http or https scheme, got {parsed.scheme!r}."
        )
    if parsed.username is not None or parsed.password is not None:
        raise SSRFBlockedError("Outbound URL must not contain embedded userinfo credentials.")

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise SSRFBlockedError(f"Outbound URL missing hostname: {raw_url!r}")
    if host in METADATA_HOSTNAMES:
        raise SSRFBlockedError(f"Outbound URL targets blocked metadata host {host!r}.")
    if host in {"localhost", "ip6-localhost", "ip6-loopback"} or host.endswith(".localhost"):
        raise SSRFBlockedError(f"Outbound URL targets loopback host {host!r}.")

    try:
        ip_literal = ipaddress.ip_address(host)
    except ValueError:
        ip_literal = None
    if ip_literal is not None:
        validate_ip_address(
            ip_literal, raw_host=host, allow_private_networks=allow_private_networks
        )
    return cleaned


def _normalized_origin(parsed: SplitResult) -> tuple[str, str, int]:
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    port = parsed.port or (443 if scheme == "https" else 80)
    return (scheme, host, port)


def validate_same_origin_url(
    candidate_url: str,
    configured_base_url: str,
    *,
    allowed_extra_hosts: frozenset[str] = frozenset(),
    allow_private_networks: bool = True,
) -> str:
    """Validates that a webhook/pagination URL shares the configured connector origin."""
    clean_candidate = validate_url_target(
        candidate_url, allow_private_networks=allow_private_networks
    )
    clean_base = validate_url_target(
        configured_base_url, allow_private_networks=allow_private_networks
    )
    cand_origin = _normalized_origin(urlsplit(clean_candidate))
    base_origin = _normalized_origin(urlsplit(clean_base))
    if cand_origin != base_origin:
        if (
            cand_origin[0] == base_origin[0]
            and cand_origin[1] in allowed_extra_hosts
            and cand_origin[2] == base_origin[2]
        ):
            return clean_candidate
        raise SSRFBlockedError(
            f"Untrusted URL origin {cand_origin[0]}://{cand_origin[1]}:{cand_origin[2]} "
            f"does not match configured connector origin "
            f"{base_origin[0]}://{base_origin[1]}:{base_origin[2]}"
        )
    return clean_candidate


class _SSRFPinnedSyncBackend(httpcore.NetworkBackend):
    """Connect-time DNS resolving & IP-pinning httpcore NetworkBackend eliminating TOCTOU."""

    def __init__(
        self,
        *,
        allow_private_networks: bool = False,
        inner: httpcore.NetworkBackend | None = None,
    ) -> None:
        self._allow_private_networks = allow_private_networks
        self._inner = inner or httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        clean_host = host.strip().lower().rstrip(".")
        if (
            clean_host in METADATA_HOSTNAMES
            or clean_host
            in {
                "localhost",
                "ip6-localhost",
                "ip6-loopback",
            }
            or clean_host.endswith(".localhost")
        ):
            raise SSRFBlockedError(f"Connect-time SSRF guard blocked host {host!r}")

        pinned_ip: str | None = None
        try:
            ip_lit = ipaddress.ip_address(clean_host)
        except ValueError:
            ip_lit = None

        if ip_lit is not None:
            safe_ip = validate_ip_address(
                ip_lit,
                raw_host=clean_host,
                allow_private_networks=self._allow_private_networks,
            )
            pinned_ip = str(safe_ip)
        else:
            addr_infos = socket.getaddrinfo(clean_host, port, type=socket.SOCK_STREAM)
            if not addr_infos:
                raise SSRFBlockedError(
                    f"Connect-time DNS resolution returned no records for {host!r}"
                )
            validated_ips: list[str] = []
            for info in addr_infos:
                raw_addr = str(info[4][0]).split("%", 1)[0]
                safe_ip = validate_ip_address(
                    raw_addr,
                    raw_host=clean_host,
                    allow_private_networks=self._allow_private_networks,
                )
                validated_ips.append(str(safe_ip))
            pinned_ip = validated_ips[0]

        return self._inner.connect_tcp(
            pinned_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        raise SSRFBlockedError(f"Unix domain socket connections are blocked (path={path!r})")

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


class SSRFPinnedHTTPTransport(httpx.BaseTransport):
    """Synchronous transport pinning connect-time DNS resolution and blocking 3xx redirects."""

    def __init__(
        self,
        *,
        allow_private_networks: bool = False,
        verify: ssl.SSLContext | bool = True,
    ) -> None:
        ssl_ctx = (
            verify
            if isinstance(verify, ssl.SSLContext)
            else (ssl.create_default_context() if verify else None)
        )
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl_ctx,
            network_backend=_SSRFPinnedSyncBackend(allow_private_networks=allow_private_networks),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        validate_url_target(
            str(request.url),
            allow_private_networks=self._pool._network_backend._allow_private_networks,  # type: ignore[attr-defined]
        )
        req = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        resp = self._pool.handle_request(req)
        return httpx.Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=resp.stream,  # type: ignore[arg-type]
            extensions=resp.extensions,
        )

    def close(self) -> None:
        self._pool.close()


def send_ssrf_safe_post(
    url: str,
    *,
    content: bytes,
    headers: dict[str, str],
    timeout_seconds: float = 5.0,
    allow_private_networks: bool = False,
    http_client: httpx.Client | None = None,
) -> httpx.Response:
    """Executes HTTP POST with DNS pinning, follow_redirects=False, and 3xx rejection."""
    clean_url = validate_url_target(url, allow_private_networks=allow_private_networks)
    if http_client is not None:
        resp = http_client.post(
            clean_url,
            content=content,
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=False,
        )
    else:
        transport = SSRFPinnedHTTPTransport(allow_private_networks=allow_private_networks)
        with httpx.Client(
            transport=transport,
            timeout=timeout_seconds,
            follow_redirects=False,
        ) as client:
            resp = client.post(clean_url, content=content, headers=headers)

    if 300 <= resp.status_code < 400:
        location = resp.headers.get("location", "")
        raise SSRFBlockedError(
            f"Outbound webhook HTTP {resp.status_code} redirect rejected (location={location!r})"
        )
    return resp


__all__ = [
    "METADATA_HOSTNAMES",
    "METADATA_IP_LITERALS",
    "SSRFBlockedError",
    "SSRFPinnedHTTPTransport",
    "send_ssrf_safe_post",
    "validate_ip_address",
    "validate_same_origin_url",
    "validate_url_target",
]
