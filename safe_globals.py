"""
Central registration for objects required by torch.load during Whisper/Emilia deserialization.

- 启动时只注册「核心」符号（torch、omegaconf、builtins），不加载 pyannote，避免 std::bad_alloc。
- 使用 Emilia（pyannote）前需调用 register_pyannote_safe_globals()。
"""

from __future__ import annotations

import builtins
import importlib
import logging
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)

# 仅 torch/omegaconf/builtins，不碰 pyannote，避免启动时内存爆掉
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

# 仅在调用 register_pyannote_safe_globals() 时导入并注册
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


def register_torch_safe_globals(
    extra_symbols: Optional[Iterable[str]] = None,
    include_pyannote: bool = False,
) -> None:
    """注册 Whisper 等所需的核心符号。默认不注册 pyannote，避免启动时加载 pyannote 导致 bad_alloc。"""
    symbols = list(CORE_SAFE_GLOBALS)
    if include_pyannote:
        symbols.extend(PYANNOTE_SAFE_GLOBALS)
    if extra_symbols:
        symbols.extend(extra_symbols)
    _register_symbols(symbols)


def register_pyannote_safe_globals() -> None:
    """注册 pyannote 相关符号，供 Emilia 加载 diarization 等模型前调用。会导入 pyannote，占用较多内存。"""
    _register_symbols(list(PYANNOTE_SAFE_GLOBALS))


# 导入时只注册核心，不加载 pyannote
register_torch_safe_globals()
