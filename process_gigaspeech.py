#!/usr/bin/env python3
"""
Process GigaSpeech 2 dataset with audio merging, UVR, and speaker diarization.
Adds speaker_id to existing transcriptions by merging short segments.

Usage:
    python process_gigaspeech.py \
        --gigaspeech-root g2_th_refined/data/th \
        --tsv-file g2_th_refined/data/th/train_refined.tsv \
        --output-dir output/gigaspeech_processed \
        --config Emilia/config.json
"""

import argparse
import json
import shutil
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 逐步导入并追踪错误
print("开始导入模块...", file=sys.stderr)

try:
    print("  [1/5] 导入基础库...", file=sys.stderr)
    import librosa
    print("    ✓ librosa", file=sys.stderr)
except Exception as e:
    print(f"    ✗ librosa 导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

try:
    import numpy as np
    print("    ✓ numpy", file=sys.stderr)
except Exception as e:
    print(f"    ✗ numpy 导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

try:
    import soundfile as sf
    print("    ✓ soundfile", file=sys.stderr)
except Exception as e:
    print(f"    ✗ soundfile 导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

try:
    from tqdm import tqdm
    print("    ✓ tqdm", file=sys.stderr)
except Exception as e:
    print(f"    ✗ tqdm 导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

try:
    print("  [2/5] 导入 safe_globals...", file=sys.stderr)
    from safe_globals import register_torch_safe_globals
    print("    ✓ safe_globals 模块导入成功", file=sys.stderr)
except Exception as e:
    print(f"    ✗ safe_globals 模块导入失败: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

try:
    print("  [3/5] 调用 register_torch_safe_globals()...", file=sys.stderr)
    register_torch_safe_globals()
    print("    ✓ register_torch_safe_globals() 调用成功", file=sys.stderr)
except Exception as e:
    print(f"    ✗ register_torch_safe_globals() 调用失败: {e}", file=sys.stderr)
    print("    ⚠ 这可能是导致 std::bad_alloc 的原因！", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

try:
    print("  [4/5] 导入 emilia_pipeline...", file=sys.stderr)
    print("    注意：这会触发 emilia_pipeline.py 的所有顶层导入", file=sys.stderr)
    from emilia_pipeline import run_emilia_pipeline
    print("    ✓ emilia_pipeline 导入成功", file=sys.stderr)
except Exception as e:
    print(f"    ✗ emilia_pipeline 导入失败: {e}", file=sys.stderr)
    print("    ⚠ 这可能是导致 std::bad_alloc 的原因！", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

print("  [5/5] 所有导入完成", file=sys.stderr)

# 全局锁，用于保护模型加载过程，避免并发加载冲突
_model_load_lock = threading.Lock()


def load_tsv(tsv_path: Path) -> Dict[str, str]:
    """
    Load TSV file and return dict mapping segment_id to text.
    
    Format: segment_id\ttext
    Example: 0-0\tสวัสดีครับ...
    """
    transcript_map = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                segment_id, text = parts
                segment_id = segment_id.strip()
                text = text.strip()
                if segment_id in transcript_map:
                    print(f"Warning: Duplicate segment_id '{segment_id}' at line {line_num}", file=sys.stderr)
                transcript_map[segment_id] = text
            else:
                print(f"Warning: Invalid line format at line {line_num}: {line[:50]}...", file=sys.stderr)
    return transcript_map


def scan_audio_groups(train_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan directory structure and group audio files by subdirectory.
    
    Groups files like train/10/10000/*.wav together.
    Returns dict mapping group_key to list of audio file paths.
    """
    groups = defaultdict(list)
    
    # Walk through train directory
    for subdir1 in train_dir.iterdir():
        if not subdir1.is_dir():
            continue
        for subdir2 in subdir1.iterdir():
            if not subdir2.is_dir():
                continue
            # Group key: e.g., "10/10000"
            group_key = f"{subdir1.name}/{subdir2.name}"
            
            # Collect all .wav files in this directory
            wav_files = sorted(subdir2.glob("*.wav"))
            if wav_files:
                groups[group_key] = wav_files
    
    return groups


def extract_segment_id_from_filename(filepath: Path) -> str:
    """Extract segment_id from filename (e.g., '10-10000-10.wav' -> '10-10000-10')."""
    return filepath.stem


def combine_audio_group(
    audio_files: List[Path],
    silence_duration: float = 0.5,
    target_sr: int = 24000,
) -> Tuple[np.ndarray, int, List[Tuple[str, float, float]]]:
    """
    Combine multiple audio files into one, with silence gaps.
    
    Args:
        audio_files: List of audio file paths, should be sorted
        silence_duration: Duration of silence between segments (seconds)
        target_sr: Target sample rate
    
    Returns:
        Tuple of (combined_waveform, sample_rate, segment_boundaries)
        segment_boundaries: list of (segment_id, start_time, end_time)
    """
    if not audio_files:
        raise ValueError("No audio files provided")
    
    segments = []
    boundaries = []
    current_time = 0.0
    
    # Load first file to get sample rate
    first_data, sr = sf.read(str(audio_files[0]), dtype="float32")
    if first_data.ndim > 1:
        first_data = librosa.to_mono(first_data.T)
    
    # Resample if needed
    if sr != target_sr:
        first_data = librosa.resample(first_data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Create silence
    silence_samples = int(sr * silence_duration)
    silence = np.zeros(silence_samples, dtype=np.float32)
    
    for audio_file in audio_files:
        segment_id = extract_segment_id_from_filename(audio_file)
        
        # Load audio
        try:
            data, file_sr = sf.read(str(audio_file), dtype="float32")
        except Exception as e:
            print(f"Error loading {audio_file}: {e}", file=sys.stderr)
            continue
        
        if data.ndim > 1:
            data = librosa.to_mono(data.T)
        
        # Resample if needed
        if file_sr != sr:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        
        # Record boundary
        duration = len(data) / sr
        boundaries.append((segment_id, current_time, current_time + duration))
        
        # Add segment
        segments.append(data)
        current_time += duration
        
        # Add silence (except after last segment)
        if audio_file != audio_files[-1]:
            segments.append(silence)
            current_time += silence_duration
    
    if not segments:
        raise ValueError("No valid audio segments to combine")
    
    # Concatenate
    combined = np.concatenate(segments)
    
    return combined, sr, boundaries


def map_emilia_segments_to_original(
    emilia_segments: List[Dict[str, Any]],
    original_boundaries: List[Tuple[str, float, float]],
) -> Dict[str, Dict[str, Any]]:
    """
    Map Emilia output segments back to original GigaSpeech segments.
    
    Args:
        emilia_segments: List of segments from Emilia output JSON
        original_boundaries: List of (segment_id, start_time, end_time) from merged audio
    
    Returns:
        Dict mapping original segment_id to Emilia segment info
    """
    mapping = {}
    
    # Create a lookup for original boundaries
    original_lookup = {seg_id: (start, end) for seg_id, start, end in original_boundaries}
    
    for emilia_seg in emilia_segments:
        emilia_start = emilia_seg.get("start", 0.0)
        emilia_end = emilia_seg.get("end", emilia_start)
        emilia_speaker = emilia_seg.get("speaker", "SPEAKER_UNKNOWN")
        emilia_text = emilia_seg.get("text", "")
        
        # Find the original segment that best matches this Emilia segment
        best_match = None
        best_overlap = 0.0
        
        for seg_id, orig_start, orig_end in original_boundaries:
            # Calculate overlap
            overlap_start = max(emilia_start, orig_start)
            overlap_end = min(emilia_end, orig_end)
            overlap = max(0.0, overlap_end - overlap_start)
            
            # Calculate overlap ratio relative to original segment
            orig_duration = orig_end - orig_start
            if orig_duration > 0:
                overlap_ratio = overlap / orig_duration
            else:
                overlap_ratio = 0.0
            
            # Prefer segments with higher overlap ratio
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_match = seg_id
        
        # If we found a good match (overlap > 50%), assign speaker
        if best_match and best_overlap > 0.5:
            if best_match not in mapping:
                mapping[best_match] = {
                    "speaker": emilia_speaker,
                    "emilia_text": emilia_text,
                    "emilia_start": emilia_start,
                    "emilia_end": emilia_end,
                    "overlap_ratio": best_overlap,
                }
            else:
                # If multiple Emilia segments match, use the one with higher overlap
                if best_overlap > mapping[best_match].get("overlap_ratio", 0.0):
                    mapping[best_match] = {
                        "speaker": emilia_speaker,
                        "emilia_text": emilia_text,
                        "emilia_start": emilia_start,
                        "emilia_end": emilia_end,
                        "overlap_ratio": best_overlap,
                    }
    
    return mapping


def load_processed_segments(jsonl_path: Path) -> set:
    """
    Load already processed segment IDs from JSONL file for resume support.
    
    Returns:
        Set of segment IDs that have already been processed
    """
    processed = set()
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            segment_id = entry.get("id")
                            if segment_id:
                                processed.add(segment_id)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Warning: Could not load processed segments from {jsonl_path}: {e}", file=sys.stderr)
    return processed


def process_single_group(
    group_key: str,
    audio_files: List[Path],
    transcript_map: Dict[str, str],
    gigaspeech_root: Path,
    output_dir: Path,
    config_path: Path,
    silence_duration: float,
    batch_size: int,
    whisper_arch: str,
    threads: int,
    do_uvr: bool,
    forced_language: str,
    group_index: int,
    total_groups: int,
    worker_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Process a single audio group: combine -> Emilia -> map -> return results.
    
    This function is designed to be called in parallel for different groups.
    """
    results = []
    
    try:
        # Create temporary directory for this group
        temp_dir = output_dir / "_temp" / f"group_{group_index}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        combined_audio_dir = temp_dir / "combined_audio"
        combined_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Combine audio
        audio_files = sorted(audio_files, key=lambda f: extract_segment_id_from_filename(f))
        try:
            combined_audio, sr, boundaries = combine_audio_group(
                audio_files,
                silence_duration=silence_duration,
                target_sr=24000,  # Emilia uses 24kHz
            )
            
            # Save combined audio
            combined_filename = f"group_{group_key.replace('/', '_')}.wav"
            combined_path = combined_audio_dir / combined_filename
            sf.write(str(combined_path), combined_audio, sr)
            
        except Exception as e:
            print(f"Error combining group {group_key}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return results
        
        # Step 2: Process with Emilia pipeline
        try:
            # 延迟导入：只在真正需要时才加载重型模型
            from safe_globals import register_torch_safe_globals
            register_torch_safe_globals()
            from emilia_pipeline import run_emilia_pipeline
            
            # 使用共享的模型缓存目录，通过锁保护模型加载过程
            # 这样模型只需要下载一次，所有 worker 可以复用
            # 注意：run_emilia_pipeline 每次调用都会重新加载模型（通过 prepare_models）
            # 我们使用锁来保护模型加载过程，避免并发加载时的文件冲突
            # 虽然这会在模型加载时串行化，但模型加载通常很快，实际推理可以并发
            with _model_load_lock:
                emilia_results = run_emilia_pipeline(
                    config_path,
                    input_folder=str(combined_audio_dir),
                    batch_size=batch_size,
                    compute_type="float16",
                    whisper_arch=whisper_arch,
                    threads=threads,
                    do_uvr=do_uvr,
                    forced_language=forced_language,
                    emilia_keep_processed=False,  # Clean up intermediate files
                )
        except Exception as e:
            print(f"Error running Emilia pipeline for group {group_key}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return results
        
        # Step 3: Extract Emilia segments
        emilia_segments = []
        for manifest_path, segments in emilia_results:
            # Try to match with our combined filename
            manifest_dir = manifest_path.parent
            output_name = manifest_dir.name
            output_name_with_ext = f"{output_name}.wav"
            
            if output_name_with_ext == combined_filename or output_name == combined_filename.replace(".wav", ""):
                emilia_segments = segments
                break
        
        # Step 4: Map results to original segments
        segment_mapping = map_emilia_segments_to_original(emilia_segments, boundaries)
        
        # Step 5: Create output entries
        for audio_file in audio_files:
            segment_id = extract_segment_id_from_filename(audio_file)
            
            # Get transcript
            text = transcript_map.get(segment_id, "")
            
            # Get speaker info from mapping
            speaker_info = segment_mapping.get(segment_id, {})
            speaker_id = speaker_info.get("speaker", "SPEAKER_UNKNOWN")
            emilia_text = speaker_info.get("emilia_text", "")
            
            # Calculate duration
            try:
                info = sf.info(str(audio_file))
                duration = info.duration
            except Exception:
                duration = 0.0
            
            # Create relative audio path
            try:
                audio_rel_path = str(audio_file.relative_to(gigaspeech_root))
            except ValueError:
                audio_rel_path = str(audio_file)
            
            entry = {
                "id": segment_id,
                "text": text,
                "audio": audio_rel_path,
                "speaker": speaker_id,
                "duration": round(duration, 2),
                "source": audio_file.name,
                "emilia_text": emilia_text,
            }
            results.append(entry)
        
        # Clean up temporary directory for this group
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"Error processing group {group_key}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    
    return results


def process_gigaspeech(
    gigaspeech_root: Path,
    tsv_file: Path,
    output_dir: Path,
    config_path: Path,
    split: str = "train",
    max_group_duration: float = 300.0,
    silence_duration: float = 0.5,
    batch_size: int = 16,
    whisper_arch: str = "medium",
    threads: int = 4,
    do_uvr: bool = True,
    forced_language: str = "th",
    max_workers: int = 2,
) -> None:
    """
    Main processing function for GigaSpeech 2 dataset.
    
    Args:
        gigaspeech_root: Root directory of GigaSpeech data (e.g., g2_th_refined/data/th)
        tsv_file: Path to TSV file with transcripts
        output_dir: Output directory for processed results
        config_path: Path to Emilia config.json
        split: Dataset split (train, dev, test)
        max_group_duration: Maximum duration for a merged audio group (seconds)
        silence_duration: Silence duration between segments (seconds)
        batch_size: WhisperX batch size
        whisper_arch: Whisper model architecture
        threads: CPU threads for Whisper
        do_uvr: Enable UVR separation
        forced_language: Language code (e.g., "th" for Thai)
    """
    # Load transcripts
    print(f"Loading transcripts from {tsv_file}...")
    transcript_map = load_tsv(tsv_file)
    print(f"Loaded {len(transcript_map)} transcripts")
    
    # Setup directories
    train_dir = gigaspeech_root / split
    if not train_dir.exists():
        raise FileNotFoundError(f"Directory not found: {train_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}_with_speakers.jsonl"
    
    # Load already processed segments for resume support
    print(f"Checking for already processed segments...")
    processed_segments = load_processed_segments(jsonl_path)
    print(f"Found {len(processed_segments)} already processed segments")
    
    # Scan audio groups
    print(f"Scanning audio files in {train_dir}...")
    audio_groups = scan_audio_groups(train_dir)
    print(f"Found {len(audio_groups)} audio groups")
    
    # Filter out groups that are already fully processed
    groups_to_process = []
    for group_key, audio_files in audio_groups.items():
        # Check if all segments in this group are already processed
        segment_ids = [extract_segment_id_from_filename(f) for f in audio_files]
        if not all(sid in processed_segments for sid in segment_ids):
            groups_to_process.append((group_key, audio_files))
    
    print(f"Groups to process: {len(groups_to_process)} (skipping {len(audio_groups) - len(groups_to_process)} already processed)")
    
    if not groups_to_process:
        print("All groups already processed!")
        return
    
    # Create temporary directory
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # File lock for thread-safe writing
    write_lock = threading.Lock()
    
    # Open JSONL file in append mode for thread-safe writing
    jsonl_file = open(jsonl_path, "a", encoding="utf-8")
    
    try:
        # Process groups concurrently
        print(f"\nProcessing {len(groups_to_process)} groups with {max_workers} workers...")
        
        total_processed = 0
        total_segments = 0
        
        # Use ThreadPoolExecutor for concurrent processing
        # Note: Using threads instead of processes because GPU resources and model loading
        # are better shared in threads, and we need to avoid duplicating model memory
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            worker_counter = 0
            worker_counter_lock = threading.Lock()
            
            for idx, (group_key, audio_files) in enumerate(groups_to_process):
                # 为每个任务分配 worker_id（循环使用，避免创建太多缓存目录）
                with worker_counter_lock:
                    worker_id = worker_counter % max_workers
                    worker_counter += 1
                
                future = executor.submit(
                    process_single_group,
                    group_key,
                    audio_files,
                    transcript_map,
                    gigaspeech_root,
                    output_dir,
                    config_path,
                    silence_duration,
                    batch_size,
                    whisper_arch,
                    threads,
                    do_uvr,
                    forced_language,
                    idx,
                    len(groups_to_process),
                    worker_id,  # 传递 worker_id
                )
                futures[future] = (group_key, idx)
            
            # Process completed tasks and write results
            for future in as_completed(futures):
                group_key, idx = futures[future]
                try:
                    results = future.result()
                    if results:
                        # Thread-safe writing
                        with write_lock:
                            for entry in results:
                                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                jsonl_file.flush()  # Ensure data is written immediately
                        total_processed += 1
                        total_segments += len(results)
                        print(f"✓ Processed group {idx+1}/{len(groups_to_process)}: {group_key} ({len(results)} segments)", file=sys.stderr)
                    else:
                        print(f"✗ Group {idx+1}/{len(groups_to_process)}: {group_key} produced no results", file=sys.stderr)
                except Exception as e:
                    print(f"✗ Error processing group {group_key}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
        
        print(f"\nProcessing complete!")
        print(f"  Processed groups: {total_processed}/{len(groups_to_process)}")
        print(f"  Total segments: {total_segments}")
        print(f"  Output: {jsonl_path}")
    
    finally:
        # Close JSONL file
        jsonl_file.close()
        
        # 清理临时目录
        if temp_dir.exists():
            print(f"\nCleaning up temporary directory: {temp_dir}", file=sys.stderr)
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Process GigaSpeech 2 dataset with UVR and speaker diarization"
    )
    parser.add_argument(
        "--gigaspeech-root",
        type=Path,
        required=True,
        help="Root directory of GigaSpeech data (e.g., g2_th_refined/data/th)",
    )
    parser.add_argument(
        "--tsv-file",
        type=Path,
        required=True,
        help="Path to TSV file with transcripts (e.g., train_refined.tsv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Emilia/config.json"),
        help="Path to Emilia config.json (default: Emilia/config.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--max-group-duration",
        type=float,
        default=300.0,
        help="Maximum duration for a merged audio group in seconds (default: 300)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.5,
        help="Silence duration between segments in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="WhisperX batch size (default: 16)",
    )
    parser.add_argument(
        "--whisper-arch",
        type=str,
        default="medium",
        help="Whisper model architecture (default: medium)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU threads for Whisper (default: 4)",
    )
    parser.add_argument(
        "--no-uvr",
        action="store_true",
        help="Disable UVR separation",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="th",
        help="Language code for transcription (default: th)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of concurrent workers for processing groups (default: 2). "
             "Note: Higher values may cause GPU memory issues.",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.gigaspeech_root.exists():
        parser.error(f"GigaSpeech root directory not found: {args.gigaspeech_root}")
    
    if not args.tsv_file.exists():
        parser.error(f"TSV file not found: {args.tsv_file}")
    
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")
    
    process_gigaspeech(
        gigaspeech_root=args.gigaspeech_root,
        tsv_file=args.tsv_file,
        output_dir=args.output_dir,
        config_path=args.config,
        split=args.split,
        max_group_duration=args.max_group_duration,
        silence_duration=args.silence_duration,
        batch_size=args.batch_size,
        whisper_arch=args.whisper_arch,
        threads=args.threads,
        do_uvr=not args.no_uvr,
        forced_language=args.language,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()

