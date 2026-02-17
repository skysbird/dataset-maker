"""
Central registration for torch.load during Whisper/Emilia deserialization.

- 导入时只注册「核心」符号（不包含 pyannote）。解析 pyannote 会触发 pyannote → torchcodec
  → load_library，在当前环境下易触发 std::bad_alloc，故默认不注册。
- 使用 Emilia 加载 pyannote 模型前，需调用 register_pyannote_safe_globals() 再 import pyannote。

调试：  python safe_globals.py  只跑核心符号并打印 RSS；不解析 pyannote。
"""

from __future__ import annotations

import builtins
import importlib
import logging
import sys
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)

# 仅 torch/omegaconf/builtins，解析时不会导入 pyannote（从而不拉 torchcodec）
CORE_SAFE_GLOBALS = [
    "omegaconf.listconfig.ListConfig",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.base.ContainerMetadata",
    "omegaconf.base.Metadata",
    "omegaconf.nodes.AnyNode",
    "omegaconf.omegaconf.OmegaConf",
    "torch.torch_version.TorchVersion",
    "typing.Any",
    "collections.defaultdict",
    "builtins.list",
    "builtins.dict",
    "builtins.int",
]

# 仅在 register_pyannote_safe_globals() 中解析；会导入 pyannote → torchcodec，可能 std::bad_alloc
PYANNOTE_SAFE_GLOBALS = [
    "pyannote.audio.core.task.Specifications",
    "pyannote.audio.core.task.Problem",
    "pyannote.audio.core.task.Resolution",
    "pyannote.audio.core.model.Introspection",
]


def _resolve_symbol(qualname: str):
    module_name, attr_name = qualname.rsplit(".", 1)
    if module_name == "builtins":
        return getattr(builtins, attr_name)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _register_symbols(symbols: list[str]) -> None:
    for qualname in symbols:
        try:
            obj = _resolve_symbol(qualname)
        except Exception as exc:
            logger.warning("Unable to resolve %s for torch safe globals: %s", qualname, exc)
            continue
        torch.serialization.add_safe_globals([obj])


def register_torch_safe_globals(extra_symbols: Optional[Iterable[str]] = None) -> None:
    """注册 Whisper 等所需核心符号；不包含 pyannote，避免导入时触发 torchcodec/std::bad_alloc。"""
    symbols = list(CORE_SAFE_GLOBALS)
    if extra_symbols:
        symbols.extend(extra_symbols)
    _register_symbols(symbols)


def register_pyannote_safe_globals() -> None:
    """注册 pyannote 相关符号（会 import pyannote → torchcodec，可能 std::bad_alloc）。供 Emilia 在加载 diarization 前调用。"""
    _register_symbols(list(PYANNOTE_SAFE_GLOBALS))


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
    try:
        import faulthandler
        faulthandler.enable(all_threads=True, file=sys.stderr)
    except Exception:
        pass

    print("safe_globals debug: CORE only (no pyannote)", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"rss~{_rss_mb()} MB", flush=True)

    for i, qualname in enumerate(CORE_SAFE_GLOBALS, 1):
        print(f"[{i}/{len(CORE_SAFE_GLOBALS)}] {qualname} ... ", end="", flush=True)
        obj = _resolve_symbol(qualname)
        torch.serialization.add_safe_globals([obj])
        print(f"OK (rss~{_rss_mb()} MB)", flush=True)

    print("safe_globals debug: CORE all OK.", flush=True)
    print("pyannote 符号未注册（会拉 torchcodec，易 bad_alloc）；Emilia 加载 diarization 前调用 register_pyannote_safe_globals()。", flush=True)
else:
    # 被 import 时只注册核心，不碰 pyannote
    register_torch_safe_globals()
