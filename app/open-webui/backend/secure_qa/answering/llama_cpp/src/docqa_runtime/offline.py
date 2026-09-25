"""Offline / loopback guarantees and the evidence we collect for them."""

from __future__ import annotations

import ipaddress
import socket

import psutil

from .errors import PortInUseError, SecurityError

# Well-known public endpoints used only to *test* whether the machine can reach
# the internet. Nothing is sent; we only try to open a TCP connection.
PROBES = [("1.1.1.1", 443), ("8.8.8.8", 53), ("9.9.9.9", 443)]


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def require_loopback(host: str, allow_remote: bool) -> None:
    if not is_loopback(host) and not allow_remote:
        raise SecurityError(
            "remote_bind_refused",
            f"Refusing to listen on {host}: the runtime only serves this computer (127.0.0.1) by default",
            "Use --host 127.0.0.1. If you really need LAN access, set allow_remote = true - "
            "this exposes the model and any document text sent to it to the network.",
        )


def outbound_network(timeout: float = 1.5) -> dict:
    """Try to open TCP connections to public IPs. Returns evidence dict."""
    attempts = []
    reachable = False
    for host, port in PROBES:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                attempts.append({"target": f"{host}:{port}", "result": "connected"})
                reachable = True
                break
        except OSError as e:
            attempts.append({"target": f"{host}:{port}", "result": type(e).__name__})
    return {"outbound_reachable": reachable, "attempts": attempts}


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET6 if ":" in host.strip("[]") else socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host.strip("[]"), port))
            return True
        except OSError:
            return False


def ensure_port_free(host: str, port: int, what: str, flag: str) -> None:
    if not port_free(host, port):
        raise PortInUseError(
            "port_in_use", f"Port {port} on {host} is already in use ({what})",
            f"Another copy of the runtime may already be running - check http://{host}:{port}/health - "
            f"or choose another port with {flag}.",
        )


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def socket_audit(pid: int) -> dict:
    """List the sockets a process holds and flag anything non-loopback."""
    try:
        proc = psutil.Process(pid)
        conns = proc.net_connections(kind="inet") if hasattr(proc, "net_connections") else proc.connections(kind="inet")
    except psutil.AccessDenied:
        return {"available": False, "reason": "access denied (run as the same user / admin to audit sockets)"}
    except psutil.Error as e:
        return {"available": False, "reason": str(e)}
    listening, remote, violations = [], [], []
    for c in conns:
        la = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "?"
        if c.status == psutil.CONN_LISTEN:
            listening.append(la)
            if not is_loopback(c.laddr.ip):
                violations.append(f"listening on non-loopback {la}")
        elif c.raddr:
            ra = f"{c.raddr.ip}:{c.raddr.port}"
            remote.append(f"{la} -> {ra} ({c.status})")
            if not is_loopback(c.raddr.ip):
                violations.append(f"connection to non-loopback {ra}")
    return {"available": True, "listening": listening, "connections": remote,
            "loopback_only": not violations, "violations": violations}
