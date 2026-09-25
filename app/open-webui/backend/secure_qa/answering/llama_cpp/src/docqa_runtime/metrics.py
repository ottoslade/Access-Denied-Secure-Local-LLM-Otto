"""Process memory measurement and machine description."""

from __future__ import annotations

import os
import platform
import threading
import time

import psutil


def native_peak_rss(pid: int) -> int | None:
    """OS-reported lifetime peak resident memory of a process, if available.

    Windows: PeakWorkingSetSize. Linux: VmHWM. Other: None (use sampling).
    """
    try:
        if os.name == "nt":
            return int(psutil.Process(pid).memory_info().peak_wset)
        status = f"/proc/{pid}/status"
        if os.path.exists(status):
            with open(status) as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024
    except (psutil.Error, OSError, AttributeError, ValueError):
        pass
    return None


def current_rss(pid: int) -> int | None:
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except psutil.Error:
        return None


class PeakSampler:
    """Samples RSS in the background; combines with the OS peak counter.

    Sampling alone can miss short spikes, the OS counter alone is not portable;
    we report the max of both and say which source won.
    """

    def __init__(self, pid: int, interval: float = 0.1):
        self.pid = pid
        self.interval = interval
        self.sampled_peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._last_native: int | None = None

    def start(self) -> "PeakSampler":
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = current_rss(self.pid)
            if rss is None:
                break
            self.sampled_peak = max(self.sampled_peak, rss)
            nat = native_peak_rss(self.pid)
            if nat:
                self._last_native = nat
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)

    def peak(self) -> tuple[int, str]:
        nat = native_peak_rss(self.pid) or self._last_native
        if nat and nat >= self.sampled_peak:
            return nat, "os_peak_counter"
        return self.sampled_peak, "sampled_rss"


def cpu_name() -> str:
    name = ""
    try:
        if os.name == "nt":
            import winreg  # type: ignore

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        elif os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
        elif platform.system() == "Darwin":
            import subprocess

            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    except Exception:
        pass
    return (name or platform.processor() or platform.machine()).strip()


def system_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": vm.total,
        "ram_available_bytes": vm.available,
        "python": platform.python_version(),
        "hostname_hash": hex(hash(platform.node()) & 0xFFFFFF),  # identify the machine without leaking its name
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MiB"
    return f"{n / 1024:.0f} KiB"
