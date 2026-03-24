from __future__ import annotations

from fedbench.runtime.platform_info import collect_platform_info

EXPECTED_KEYS = {
    "platform.os",
    "platform.os_version",
    "platform.cpu_model",
    "platform.cpu_count",
    "platform.ram_gb",
    "platform.python_version",
    "platform.torch_version",
    "platform.flwr_version",
    "platform.numpy_version",
    "platform.sklearn_version",
    "platform.gpu_model",
}


def test_returns_all_expected_keys() -> None:
    info = collect_platform_info()
    assert set(info.keys()) == EXPECTED_KEYS


def test_value_types() -> None:
    info = collect_platform_info()
    for key, value in info.items():
        assert isinstance(value, (str, int, type(None))), (
            f"{key} has unexpected type {type(value)}"
        )


def test_python_version_starts_with_3() -> None:
    info = collect_platform_info()
    assert isinstance(info["platform.python_version"], str)
    assert info["platform.python_version"].startswith("3.")


def test_os_is_nonempty_string() -> None:
    info = collect_platform_info()
    assert isinstance(info["platform.os"], str)
    assert len(info["platform.os"]) > 0


def test_cpu_count_is_positive_or_none() -> None:
    info = collect_platform_info()
    cpu_count = info["platform.cpu_count"]
    assert cpu_count is None or (isinstance(cpu_count, int) and cpu_count > 0)


def test_ram_gb_is_positive_or_none() -> None:
    info = collect_platform_info()
    ram = info["platform.ram_gb"]
    assert ram is None or (isinstance(ram, int) and ram > 0)
