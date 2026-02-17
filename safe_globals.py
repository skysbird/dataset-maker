"""
Central registration for objects required by torch.load during Whisper/Emilia deserialization.

调试方式（仅针对本文件）：
    python safe_globals.py

会逐个符号 resolve + 注册，并打印进度与当前 RSS（尽量定位触发 std::bad_alloc 的具体符号/模块）。
"""

from __future__ import annotations

import builtins
import importlib
import logging
import sys
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_SAFE_GLOBALS = [
    "omegaconf.listconfig.ListConfig",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.base.ContainerMetadata",
    "omegaconf.base.Metadata",
    "omegaconf.nodes.AnyNode",
    "omegaconf.omegaconf.OmegaConf",
    "torch.torch_version.TorchVersion",
    "pyannote.audio.core.task.Specifications",
    "pyannote.audio.core.task.Problem",
    "pyannote.audio.core.task.Resolution",
    "pyannote.audio.core.model.Introspection",
    "typing.Any",
    "collections.defaultdict",
    "builtins.list",
    "builtins.dict",
    "builtins.int",
]


def _resolve_symbol(qualname: str):
    module_name, attr_name = qualname.rsplit(".", 1)
    if module_name == "builtins":
        return getattr(builtins, attr_name)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def register_torch_safe_globals(extra_symbols: Optional[Iterable[str]] = None) -> None:
    symbols = list(DEFAULT_SAFE_GLOBALS)
    if extra_symbols:
        symbols.extend(extra_symbols)

    for qualname in symbols:
        try:
            obj = _resolve_symbol(qualname)
        except Exception as exc:
            logger.warning("Unable to resolve %s for torch safe globals: %s", qualname, exc)
            continue
        torch.serialization.add_safe_globals([obj])


def _rss_mb() -> int:
    """Best-effort RSS in MB (Linux preferred)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: KB, macOS: bytes. 这里按 Linux 优先处理；macOS 下值会偏大但仍可用作趋势。
        v = int(getattr(r, "ru_maxrss", 0))
        return v // 1024 if v > 10_000 else v // (1024 * 1024)
    except Exception:
        try:
            import psutil

            return psutil.Process().memory_info().rss // (1024 * 1024)
        except Exception:
            return 0


if __name__ == "__main__":
    # 只在直接运行时做“逐步调试”，避免一上来就把所有依赖拉起导致无法定位
    try:
        import faulthandler

        faulthandler.enable(all_threads=True, file=sys.stderr)
    except Exception:
        pass

    print("safe_globals debug: start", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"rss~{_rss_mb()} MB", flush=True)

    # 逐个符号 resolve + register，崩溃时最后一行就是触发点
    for i, qualname in enumerate(DEFAULT_SAFE_GLOBALS, 1):
        print(f"[{i}/{len(DEFAULT_SAFE_GLOBALS)}] {qualname} ... ", end="", flush=True)
        obj = _resolve_symbol(qualname)  # 若这里触发 std::bad_alloc，会直接中止
        torch.serialization.add_safe_globals([obj])
        print(f"OK (rss~{_rss_mb()} MB)", flush=True)

    print("safe_globals debug: all OK", flush=True)
else:
    # 保持原语义：被其他模块 import 时自动注册
    register_torch_safe_globals()
