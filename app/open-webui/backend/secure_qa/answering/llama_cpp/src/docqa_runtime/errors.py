"""Actionable errors.

Every failure the user can hit carries a stable ``code``, a plain-language
``message`` and a ``hint`` telling them what to do next. The CLI prints all
three; the gateway returns them as an OpenAI-style error body.
"""

from __future__ import annotations

# Exit codes are part of the contract (scripts / E5 test harness rely on them).
EXIT_OK = 0
EXIT_CONFIG = 2          # bad config / flags / missing files
EXIT_BACKEND = 3         # llama-server failed to start or crashed
EXIT_PORT = 4            # port already in use
EXIT_RESOURCES = 5       # not enough RAM for the chosen model
EXIT_SECURITY = 6        # refused because it would break offline/loopback guarantees


class RuntimeFailure(Exception):
    exit_code = EXIT_CONFIG
    http_status = 500

    def __init__(self, code: str, message: str, hint: str = "", *, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message, "hint": self.hint}
        if self.details:
            d["details"] = self.details
        return d

    def render(self) -> str:
        lines = [f"ERROR [{self.code}] {self.message}"]
        if self.hint:
            lines.append(f"  -> {self.hint}")
        for k, v in self.details.items():
            if k == "log_tail" and v:
                lines.append("  last backend log lines:")
                lines.extend(f"    | {ln}" for ln in v)
            elif k != "log_tail":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class ConfigError(RuntimeFailure):
    exit_code = EXIT_CONFIG


class BackendError(RuntimeFailure):
    exit_code = EXIT_BACKEND
    http_status = 503


class PortInUseError(RuntimeFailure):
    exit_code = EXIT_PORT


class ResourceError(RuntimeFailure):
    exit_code = EXIT_RESOURCES


class SecurityError(RuntimeFailure):
    exit_code = EXIT_SECURITY
