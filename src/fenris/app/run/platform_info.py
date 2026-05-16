"""Collect hardware and software metadata for metrics.json."""

from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


def collect_platform_info() -> dict[str, str | int | None]:
    """Return hardware/software metadata.

    Every field is best-effort: unsupported platforms produce ``None``
    rather than raising.  No external dependencies are required.
    """
    return {
        "platform.os": platform.system(),
        "platform.os_version": platform.release(),
        "platform.cpu_model": _cpu_model(),
        "platform.cpu_count": os.cpu_count(),
        "platform.gpu_model": _gpu_model(),
        "platform.ram_gb": _ram_gb(),
        "platform.python_version": platform.python_version(),
        "platform.torch_version": _optional_version("torch"),
        "platform.flwr_version": _optional_version("flwr"),
        "platform.numpy_version": _optional_version("numpy"),
        "platform.sklearn_version": _optional_version("scikit-learn"),
    }


# platform.processor() returns bare arch strings on many Linux distros / WSL.
_ARCH_ONLY = {"x86_64", "i386", "i686", "aarch64", "arm64", "armv7l", "AMD64"}


def _cpu_model() -> str | None:
    # /proc/cpuinfo gives the real model name on Linux (incl. WSL).
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # Fallback: platform.processor(), but only if it's a real model string.
    proc = platform.processor()
    if proc and proc not in _ARCH_ONLY:
        return proc
    return None


def _ram_gb() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return round(pages * page_size / (1024**3))
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _gpu_model() -> str | None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except (ImportError, RuntimeError):
        pass
    return None


def _optional_version(package: str) -> str | None:
    if package == "torch":
        try:
            import torch  # noqa: PLC0415

            return str(torch.__version__)
        except ImportError:
            return None
    try:
        return _pkg_version(package)
    except PackageNotFoundError:
        return None
