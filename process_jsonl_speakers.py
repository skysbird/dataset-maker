#!/usr/bin/env python3
"""
处理 JSONL 格式数据集：在每条记录上跑说话人分离（Emilia），为每条添加 speaker 字段。

输入 JSONL 格式（每行一条）：
  {"id": 1, "text": "...", "audio": "output-dataset/audio/1.wav", "language": "th", "duration": 3.24, "source": "1.wav"}

输出：同格式，增加 "speaker" 字段（及可选 "emilia_text"）。

用法：
    python process_jsonl_speakers.py \
        --input output-dataset/manifest.jsonl \
        --audio-root . \
        --output output-dataset/manifest_with_speakers.jsonl \
        --config Emilia/config.json
"""

import argparse
import json
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保脚本所在目录在 sys.path 中，便于从任意工作目录运行时找到 safe_globals、emilia_pipeline 等
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

print("开始导入模块...", file=sys.stderr)
try:
    import librosa
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm
except Exception as e:
    print(f"导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

try:
    from safe_globals import register_torch_safe_globals
    register_torch_safe_globals()
    from emilia_pipeline import run_emilia_pipeline
except Exception as e:
    print(f"emilia 相关导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

# 从 process_gigaspeech 复用说话人映射逻辑
from process_gigaspeech import map_emilia_segments_to_original

_model_load_lock = threading.Lock()


def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL，每行一个 dict。"""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: 第 {line_num} 行 JSON 解析失败: {e}", file=sys.stderr)
    return rows


def resolve_audio_path(entry: Dict[str, Any], audio_root: Path) -> Path:
    """根据 audio_root 解析 entry 中的 audio 路径。"""
    raw = entry.get("audio", "")
    if not raw:
        raise ValueError(f"entry 缺少 audio: id={entry.get('id')}")
    p = Path(raw)
    if p.is_absolute():
        return p
    return (audio_root / raw).resolve()


def combine_audio_group_from_entries(
    entries: List[Dict[str, Any]],
    audio_root: Path,
    silence_duration: float = 0.5,
    target_sr: int = 24000,
) -> Tuple[np.ndarray, int, List[Tuple[str, float, float]]]:
    """
    按 entries 顺序把多条音频拼成一条（中间加静音），并返回每条的时间边界。
    boundaries 中 segment_id 使用 str(entry["id"])。
    """
    if not entries:
        raise ValueError("entries 为空")

    segments = []
    boundaries = []
    current_time = 0.0
    sr = target_sr
    silence_samples = None

    for i, entry in enumerate(entries):
        segment_id = str(entry.get("id", i))
        try:
            path = resolve_audio_path(entry, audio_root)
        except ValueError as e:
            print(f"Warning: 跳过条目 id={segment_id}: {e}", file=sys.stderr)
            continue

        if not path.exists():
            print(f"Warning: 文件不存在，跳过 id={segment_id}: {path}", file=sys.stderr)
            continue

        try:
            data, file_sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            print(f"Warning: 读取失败 id={segment_id} {path}: {e}", file=sys.stderr)
            continue

        if data.ndim > 1:
            data = librosa.to_mono(data.T)
        if file_sr != target_sr:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=target_sr)

        duration = len(data) / target_sr
        boundaries.append((segment_id, current_time, current_time + duration))
        segments.append(data)
        current_time += duration

        if i < len(entries) - 1:
            if silence_samples is None:
                silence_samples = int(target_sr * silence_duration)
            silence = np.zeros(silence_samples, dtype=np.float32)
            segments.append(silence)
            current_time += silence_duration

    if not segments:
        raise ValueError("没有有效音频段可拼接")

    combined = np.concatenate(segments)
    return combined, target_sr, boundaries


def group_entries_by_duration(
    entries: List[Dict[str, Any]],
    max_group_duration: float = 300.0,
) -> List[List[Dict[str, Any]]]:
    """按顺序分组，每组总 duration 不超过 max_group_duration。"""
    groups = []
    current = []
    current_duration = 0.0

    for entry in entries:
        duration = float(entry.get("duration", 0.0))
        if duration <= 0:
            duration = 3.0  # 占位
        if current and current_duration + duration > max_group_duration:
            groups.append(current)
            current = []
            current_duration = 0.0
        current.append(entry)
        current_duration += duration

    if current:
        groups.append(current)
    return groups


def load_processed_ids(jsonl_path: Path) -> set:
    """已处理过的 id 集合（用于断点续跑）。"""
    processed = set()
    if not jsonl_path.exists():
        return processed
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    vid = entry.get("id")
                    if vid is not None:
                        processed.add(str(vid))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: 无法读取已处理列表 {jsonl_path}: {e}", file=sys.stderr)
    return processed


def process_single_group_jsonl(
    group_entries: List[Dict[str, Any]],
    group_index: int,
    audio_root: Path,
    output_dir: Path,
    config_path: Path,
    silence_duration: float,
    batch_size: int,
    whisper_arch: str,
    threads: int,
    do_uvr: bool,
    forced_language: str,
    temp_base: Path,
) -> List[Dict[str, Any]]:
    """
    处理一组条目：拼接音频 -> Emilia -> 按时间映射回每条 -> 为每条写入 speaker。
    """
    results = []
    if not group_entries:
        return results

    temp_dir = temp_base / f"group_{group_index}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    combined_audio_dir = temp_dir / "combined_audio"
    combined_audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) 拼接
        combined, sr, boundaries = combine_audio_group_from_entries(
            group_entries,
            audio_root,
            silence_duration=silence_duration,
            target_sr=24000,
        )
        combined_filename = f"group_{group_index}.wav"
        combined_path = combined_audio_dir / combined_filename
        sf.write(str(combined_path), combined, sr)

        if not combined_path.exists() or combined_path.stat().st_size == 0:
            print(f"Error: 拼接音频无效 group_{group_index}", file=sys.stderr)
            return results

        # 2) Emilia
        with _model_load_lock:
            emilia_results = run_emilia_pipeline(
                str(config_path),
                input_folder=str(combined_audio_dir.resolve()),
                batch_size=batch_size,
                compute_type="float16",
                whisper_arch=whisper_arch,
                threads=threads,
                do_uvr=do_uvr,
                forced_language=forced_language,
                emilia_keep_processed=False,
            )

        emilia_segments = []
        output_name_prefix = combined_filename.replace(".wav", "")

        if emilia_results:
            for manifest_path, segments in emilia_results:
                name = manifest_path.parent.name
                if name == output_name_prefix or f"group_{group_index}" in name:
                    emilia_segments = segments
                    output_name_prefix = name
                    break
            if not emilia_segments and len(emilia_results) == 1:
                _, emilia_segments = emilia_results[0]
                output_name_prefix = emilia_results[0][0].parent.name

        # 3) 映射回原始 segment
        segment_mapping = map_emilia_segments_to_original(
            emilia_segments,
            boundaries,
            output_name_prefix=output_name_prefix or f"group_{group_index}",
        )

        # 4) 为每条 entry 生成输出（保留原字段 + speaker）
        for entry in group_entries:
            segment_id = str(entry.get("id", ""))
            info = segment_mapping.get(segment_id, {})
            speaker_id = info.get("speaker")
            if not speaker_id:
                print(f"Warning: 未找到说话人 id={segment_id}，仍写入 speaker=UNKNOWN", file=sys.stderr)
                speaker_id = "UNKNOWN"

            out = dict(entry)
            out["speaker"] = speaker_id
            if info.get("emilia_text") is not None:
                out["emilia_text"] = info.get("emilia_text", "")
            results.append(out)

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def process_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    audio_root: Path,
    config_path: Path,
    max_group_duration: float = 300.0,
    silence_duration: float = 0.5,
    batch_size: int = 16,
    whisper_arch: str = "medium",
    threads: int = 4,
    do_uvr: bool = True,
    forced_language: str = "th",
    resume: bool = True,
) -> None:
    """
    主流程：读 JSONL -> 按时长分组 -> 每组拼接+Emilia+映射 -> 写带 speaker 的 JSONL。
    """
    rows = load_jsonl(input_jsonl)
    if not rows:
        print("没有有效行")
        return

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(output_jsonl) if resume else set()
    groups = group_entries_by_duration(rows, max_group_duration=max_group_duration)

    # 只处理至少有一条未处理 id 的组（整组会一起跑 Emilia，但只写入未处理的 id）
    groups_to_do = []
    for g in groups:
        ids_in_group = [str(e.get("id")) for e in g if e.get("id") is not None]
        if not all(sid in processed_ids for sid in ids_in_group):
            groups_to_do.append(g)

    if not groups_to_do:
        print("所有组均已处理完毕")
        return

    project_root = Path(__file__).parent.resolve()
    temp_base = project_root / "jsonl_speakers_temp"
    temp_base.mkdir(parents=True, exist_ok=True)

    mode = "a" if resume and output_jsonl.exists() else "w"
    with open(output_jsonl, mode, encoding="utf-8") as out_f:
        for idx, group_entries in enumerate(tqdm(groups_to_do, desc="处理分组")):
            try:
                results = process_single_group_jsonl(
                    group_entries,
                    idx,
                    audio_root=audio_root,
                    output_dir=output_jsonl.parent,
                    config_path=config_path,
                    silence_duration=silence_duration,
                    batch_size=batch_size,
                    whisper_arch=whisper_arch,
                    threads=threads,
                    do_uvr=do_uvr,
                    forced_language=forced_language,
                    temp_base=temp_base,
                )
                for entry in results:
                    eid = str(entry.get("id", ""))
                    if resume and eid in processed_ids:
                        continue
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed_ids.add(eid)
            except Exception as e:
                print(f"处理分组 {idx} 失败: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    if temp_base.exists():
        shutil.rmtree(temp_base, ignore_errors=True)
    print(f"完成，输出: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="对 JSONL 数据集做说话人分离，为每条添加 speaker 字段",
    )
    parser.add_argument("--input", type=Path, required=True, help="输入 JSONL 路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL 路径（含 speaker）")
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("."),
        help="audio 字段相对路径的根目录（默认当前目录）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Emilia/config.json"),
        help="Emilia config.json 路径",
    )
    parser.add_argument("--max-group-duration", type=float, default=300.0, help="每组最大时长(秒)")
    parser.add_argument("--silence-duration", type=float, default=0.5, help="拼接时静音间隔(秒)")
    parser.add_argument("--batch-size", type=int, default=16, help="WhisperX batch size")
    parser.add_argument("--whisper-arch", type=str, default="medium", help="Whisper 模型")
    parser.add_argument("--threads", type=int, default=4, help="CPU 线程数")
    parser.add_argument("--no-uvr", action="store_true", help="关闭 UVR")
    parser.add_argument("--language", type=str, default="th", help="语言代码")
    parser.add_argument("--no-resume", action="store_true", help="不续跑，从头写输出文件")

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"输入文件不存在: {args.input}")
    if not args.config.exists():
        parser.error(f"配置文件不存在: {args.config}")

    process_jsonl(
        input_jsonl=args.input,
        output_jsonl=args.output,
        audio_root=args.audio_root.resolve(),
        config_path=args.config.resolve(),
        max_group_duration=args.max_group_duration,
        silence_duration=args.silence_duration,
        batch_size=args.batch_size,
        whisper_arch=args.whisper_arch,
        threads=args.threads,
        do_uvr=not args.no_uvr,
        forced_language=args.language,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
