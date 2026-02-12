#!/usr/bin/env python3
"""
从 JSONL 数据集中筛选高质量音频，输出新的 JSONL。

质量要求：
- 说话人无重叠：单段内仅一人说话（可选，需 Hugging Face token 做 diarization）
- 无杂音、纯净人声：使用 DNSMOS 的 SIG（信号）/ BAK（背景）过滤

用法：
    uv run python filter_quality_jsonl.py \\
        --input-jsonl path/to/input.jsonl \\
        --output-jsonl path/to/output_quality.jsonl \\
        --target-hours 10 \\
        [--audio-root path/to/root] \\
        [--config Emilia/config.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm

# Optional: pyannote for diarization
try:
    import pyannote
    from pyannote.audio import Pipeline
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False

# DNSMOS from Emilia
from Emilia.models import dnsmos


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL，yield 每条记录。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def resolve_audio_path(audio_key: str, audio_root: Path) -> Path:
    """将 JSONL 中的 audio 字段解析为绝对路径。"""
    p = Path(audio_key)
    if p.is_absolute():
        return p
    return (audio_root / audio_key).resolve()


def load_audio_for_dnsmos(audio_path: Path, sr: int = 16000) -> Optional[np.ndarray]:
    """加载单声道 16k 音频供 DNSMOS 使用。"""
    if not audio_path.exists():
        return None
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        return None


def load_audio_dict(audio_path: Path, target_sr: int = 16000) -> Optional[Dict[str, Any]]:
    """加载为 emilia 风格的 audio dict，供 diarization 使用。"""
    if not audio_path.exists():
        return None
    try:
        y, orig_sr = librosa.load(str(audio_path), sr=None, mono=True)
        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return {
            "waveform": y.astype(np.float32),
            "sample_rate": target_sr,
            "name": audio_path.name,
        }
    except Exception:
        return None


def count_speakers_in_segment(diarisation: Any, audio: Dict[str, Any], device: torch.device) -> int:
    """对整段音频做 diarization，返回出现的说话人数量。若为 0 表示出错或无语音。"""
    if not HAS_PYANNOTE or diarisation is None:
        return 1  # 不检查时视为单说话人通过
    waveform = torch.tensor(audio["waveform"], device=device).unsqueeze(0)
    diar = diarisation(
        {"waveform": waveform, "sample_rate": audio["sample_rate"], "channel": 0}
    )
    if hasattr(diar, "itertracks"):
        annotation = diar
    elif hasattr(diar, "speaker_diarization"):
        annotation = diar.speaker_diarization
    else:
        return 0
    speakers = set()
    for _track, _segment, label in annotation.itertracks(yield_label=True):
        speakers.add(label)
    return len(speakers)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 JSONL 中筛选高质量音频（无说话人重叠、低杂音、纯净人声），按目标小时数截断并输出新 JSONL。"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="输入 JSONL 路径（每行一条：id, text, audio, speaker, language, duration, source 等）",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="输出高质量条目的 JSONL 路径",
    )
    parser.add_argument(
        "--target-hours",
        type=float,
        default=10.0,
        help="达到该小时数后停止筛选（默认 10）",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default=".",
        help="音频路径相对于此目录解析（默认当前目录）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Emilia 配置文件路径（用于 DNSMOS 模型路径与可选 diarization）。若不提供则需 --dnsmos-model",
    )
    parser.add_argument(
        "--dnsmos-model",
        type=str,
        default=None,
        help="DNSMOS ONNX 模型路径（例如 emilia_models/sig_bak_ovr.onnx）。若提供 --config 则可从 config 读取",
    )
    parser.add_argument(
        "--min-sig",
        type=float,
        default=3.0,
        help="DNSMOS 信号 (SIG) 下限，越高越要求人声清晰（默认 3.0）",
    )
    parser.add_argument(
        "--max-bak",
        type=float,
        default=2.5,
        help="DNSMOS 背景 (BAK) 上限，越低越要求无杂音（默认 2.5）",
    )
    parser.add_argument(
        "--min-ovrl",
        type=float,
        default=2.8,
        help="DNSMOS 总体 (OVRL) 下限（默认 2.8）",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="不做说话人重叠检测（仅用 DNSMOS 过滤）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="运行设备",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    audio_root = Path(args.audio_root).resolve()
    target_seconds = args.target_hours * 3600.0

    if not input_path.exists():
        print(f"Error: 输入文件不存在: {input_path}", file=sys.stderr)
        return 1

    # 解析 DNSMOS 模型路径与配置
    cfg: Optional[Dict[str, Any]] = None
    config_path = Path(args.config) if args.config else None
    dnsmos_path = args.dnsmos_model
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dnsmos_path = dnsmos_path or (cfg.get("mos_model") or {}).get("primary_model_path")
        if dnsmos_path:
            dnsmos_path = (config_path.parent / dnsmos_path).resolve()
    if not dnsmos_path:
        print("Error: 请提供 --config 或 --dnsmos-model 指定 DNSMOS 模型路径", file=sys.stderr)
        return 1
    if not Path(dnsmos_path).exists():
        print(f"Error: DNSMOS 模型不存在: {dnsmos_path}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    device_name = "cuda" if args.device == "cuda" else "cpu"
    scorer = dnsmos.ComputeScore(str(dnsmos_path), device_name)

    # 可选：加载 diarization
    diarisation = None
    if not args.no_diarization and cfg and HAS_PYANNOTE:
        hf_token = (cfg.get("huggingface_token") or os.environ.get("HF_TOKEN") or "").strip()
        if hf_token.startswith("hf"):
            cache_root = Path(cfg.get("download_cache") or config_path.parent).resolve()
            cache_root.mkdir(parents=True, exist_ok=True)
            os.environ["PYANNOTE_AUDIO_CACHE"] = str(cache_root / "pyannote")
            diarisation = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=hf_token,
                cache_dir=str(cache_root / "pyannote"),
            )
            diarisation.to(device)
            print("Diarization 已启用：将过滤多说话人重叠片段。", file=sys.stderr)
        else:
            print("未设置有效 Hugging Face token，跳过说话人重叠检测。", file=sys.stderr)
    elif args.no_diarization:
        print("已指定 --no-diarization，仅按 DNSMOS 过滤。", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_duration = 0.0
    kept = 0
    rejected_no_file = 0
    rejected_dnsmos = 0
    rejected_speakers = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in tqdm(load_jsonl(input_path), desc="Filtering"):
            if total_duration >= target_seconds:
                break

            duration = float(entry.get("duration") or 0)
            if duration <= 0:
                continue

            audio_key = entry.get("audio") or entry.get("source") or ""
            if not audio_key:
                continue
            audio_path = resolve_audio_path(audio_key, audio_root)
            if not audio_path.exists():
                rejected_no_file += 1
                continue

            # DNSMOS
            wav = load_audio_for_dnsmos(audio_path)
            if wav is None:
                rejected_no_file += 1
                continue
            try:
                scores = scorer(wav, 16000, False)
            except Exception:
                rejected_dnsmos += 1
                continue
            sig = float(scores["SIG"])
            bak = float(scores["BAK"])
            ovrl = float(scores["OVRL"])
            if sig < args.min_sig or bak > args.max_bak or ovrl < args.min_ovrl:
                rejected_dnsmos += 1
                continue

            # 可选：单说话人检查
            if diarisation is not None:
                audio_dict = load_audio_dict(audio_path, target_sr=16000)
                if audio_dict is None:
                    rejected_no_file += 1
                    continue
                n_speakers = count_speakers_in_segment(diarisation, audio_dict, device)
                if n_speakers != 1:
                    rejected_speakers += 1
                    continue

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            total_duration += duration
            kept += 1

    print(
        f"完成：保留 {kept} 条，总时长 {total_duration / 3600:.2f} 小时；"
        f" 拒绝：无文件/加载失败 {rejected_no_file}，DNSMOS 不通过 {rejected_dnsmos}，多说话人 {rejected_speakers}。",
        file=sys.stderr,
    )
    print(f"输出: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
