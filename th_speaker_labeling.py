#!/usr/bin/env python3
"""
CLI tool for Thai speaker labeling.
Processes audio files with UVR, combines by group, performs speaker diarization,
and generates JSONL output matching the webui format.

Usage:
    python th_speaker_labeling.py \
        --tsv <path_to_tsv> \
        --audio-dir <path_to_root_dir> \
        --output-dir <path_to_output> \
        --config <path_to_emilia_config> \
        [--dir-name th] \
        [--silence-duration 0.5]
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

# Import reusable functions
from process_gigaspeech import combine_audio_group, map_emilia_segments_to_original, extract_segment_id_from_filename
from emilia_pipeline import (
    separate_sources,
    diarise_speakers,
    prepare_models,
    load_cfg,
    _resolve_path,
    AudioDict,
)
from Emilia.utils.tool import check_env
from Emilia.utils.logger import Logger

# Ensure torch safe globals
import safe_globals


def load_tsv_labels(tsv_path: Path) -> Dict[str, str]:
    """
    Load TSV file with format: filename\ttext
    Returns a dictionary mapping filename_stem to text.
    """
    labels = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line {line_num}: {line}", file=sys.stderr)
                continue
            filename, text = parts
            # Remove extension if present
            filename_stem = Path(filename).stem
            labels[filename_stem] = text.strip()
    return labels


def find_audio_files(audio_dir: Path) -> List[Path]:
    """
    Recursively find all WAV files in the directory tree.
    """
    audio_files = []
    # Find both .wav and .WAV files
    audio_files.extend(audio_dir.rglob("*.wav"))
    audio_files.extend(audio_dir.rglob("*.WAV"))
    return sorted(audio_files)


def group_audio_files(audio_files: List[Path]) -> Dict[str, List[Path]]:
    """
    Group audio files by prefix (e.g., 0-0-0.wav -> group "0-0").
    Returns a dictionary mapping group_prefix to sorted list of files.
    """
    groups = {}
    for audio_file in audio_files:
        filename_stem = audio_file.stem
        # Extract prefix (first two numbers separated by dash)
        # e.g., "0-0-0" -> "0-0", "0-0-10" -> "0-0"
        parts = filename_stem.split('-')
        if len(parts) >= 2:
            group_prefix = f"{parts[0]}-{parts[1]}"
        else:
            # Fallback: use the whole stem as group
            group_prefix = filename_stem
        
        if group_prefix not in groups:
            groups[group_prefix] = []
        groups[group_prefix].append(audio_file)
    
    # Sort files within each group by the numeric suffix
    for group_prefix in groups:
        groups[group_prefix].sort(key=lambda f: _extract_suffix_number(f.stem))
    
    return groups


def _extract_suffix_number(filename_stem: str) -> int:
    """
    Extract numeric suffix from filename for sorting.
    e.g., "0-0-0" -> 0, "0-0-10" -> 10
    """
    parts = filename_stem.split('-')
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except ValueError:
            return 0
    return 0


def process_uvr_file(
    audio_file: Path,
    separator,
    output_audio_dir: Path,
    target_sample_rate: int = 24000
) -> Tuple[Path, bool, bool]:
    """
    Process a single audio file with UVR (remove background, keep vocals only).
    Returns (output_file_path, success, was_skipped).
    Note: The output file keeps the same name as the input file to ensure
    segment_id matching in boundaries.
    """
    # Keep the same filename to ensure segment_id consistency
    output_file = output_audio_dir / audio_file.name
    
    # Skip if already processed
    if output_file.exists():
        try:
            # Verify file is valid by checking it can be read
            sf.info(str(output_file))
            return output_file, True, True  # (file_path, success, was_skipped)
        except Exception:
            # File exists but may be corrupted, re-process it
            pass
    
    try:
        # Use separate_sources to process the file
        audio_dict = separate_sources(separator, str(audio_file), target_sample_rate)
        
        # Save the processed audio (vocals only)
        sf.write(
            str(output_file),
            audio_dict["waveform"],
            audio_dict["sample_rate"]
        )
        
        return output_file, True, False  # (file_path, success, was_skipped)
    except Exception as e:
        print(f"Error processing {audio_file} with UVR: {e}", file=sys.stderr)
        return None, False, False


def process_group(
    group_prefix: str,
    group_files: List[Path],
    labels: Dict[str, str],
    models: Dict[str, Any],
    output_dir: Path,
    output_audio_dir: Path,
    dir_name: str,
    silence_duration: float = 0.5,
    target_sample_rate: int = 24000,
    uvr_workers: int = 8
) -> List[Dict[str, Any]]:
    """
    Process a single group: UVR, combine, diarize, and map speakers.
    Returns a list of JSONL entries.
    """
    entries = []
    
    # Step 1: Process all files with UVR (parallel processing)
    print(f"Processing group {group_prefix}: UVR processing {len(group_files)} files...", file=sys.stderr)
    uvr_files = []
    
    # Use thread-local storage for separator to avoid conflicts
    separator_local = threading.local()
    
    def get_separator():
        """Get thread-local separator."""
        if not hasattr(separator_local, 'separator'):
            separator_local.separator = models["separator"]
        return separator_local.separator
    
    def process_single_uvr(audio_file: Path) -> Tuple[Path, bool, bool]:
        """Process a single file with UVR, returns (output_file, success, was_skipped)."""
        separator = get_separator()
        return process_uvr_file(
            audio_file,
            separator,
            output_audio_dir,
            target_sample_rate
        )
    
    # Process files in parallel using ThreadPoolExecutor
    max_workers = min(uvr_workers, len(group_files))
    skipped_count = 0
    processed_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_uvr, audio_file): audio_file 
                   for audio_file in group_files}
        
        # Process completed tasks with progress indication
        for future in tqdm(as_completed(futures), total=len(group_files), 
                         desc=f"UVR {group_prefix}", leave=False, file=sys.stderr):
            try:
                uvr_file, success, was_skipped = future.result()
                if success and uvr_file:
                    if was_skipped:
                        skipped_count += 1
                    else:
                        processed_count += 1
                    uvr_files.append(uvr_file)
                else:
                    failed_count += 1
            except Exception as e:
                audio_file = futures[future]
                print(f"Error processing {audio_file}: {e}", file=sys.stderr)
                failed_count += 1
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} already-processed files", file=sys.stderr)
    if processed_count > 0:
        print(f"  Processed {processed_count} new files", file=sys.stderr)
    if failed_count > 0:
        print(f"  Failed {failed_count} files", file=sys.stderr)
    
    if not uvr_files:
        print(f"Warning: No files processed for group {group_prefix}", file=sys.stderr)
        return entries
    
    # Step 2: Combine audio files
    # Note: combine_audio_group uses extract_segment_id_from_filename which returns file.stem
    # So boundaries will have segment_id matching the original filename stem
    print(f"Combining {len(uvr_files)} files for group {group_prefix}...", file=sys.stderr)
    try:
        combined_waveform, sample_rate, boundaries = combine_audio_group(
            uvr_files,
            silence_duration=silence_duration,
            target_sr=target_sample_rate
        )
    except Exception as e:
        print(f"Error combining audio for group {group_prefix}: {e}", file=sys.stderr)
        return entries
    
    # Step 3: Prepare audio dict for diarization
    audio_dict: AudioDict = {
        "waveform": combined_waveform,
        "sample_rate": sample_rate,
        "name": f"{group_prefix}_combined"
    }
    
    # Step 4: Run speaker diarization
    print(f"Running speaker diarization for group {group_prefix}...", file=sys.stderr)
    try:
        diarization_df = diarise_speakers(
            models["diarisation"],
            audio_dict,
            models["device"]
        )
    except Exception as e:
        print(f"Error in speaker diarization for group {group_prefix}: {e}", file=sys.stderr)
        return entries
    
    # Step 5: Convert DataFrame to segment list
    emilia_segments = []
    for _, row in diarization_df.iterrows():
        emilia_segments.append({
            "start": row["start"],
            "end": row["end"],
            "speaker": row["speaker"],
            "text": ""  # We don't need text from diarization
        })
    
    # Step 6: Map speakers to original files
    output_name_prefix = f"{dir_name}_{group_prefix}"
    segment_mapping = map_emilia_segments_to_original(
        emilia_segments,
        boundaries,
        output_name_prefix=output_name_prefix
    )
    
    # Step 7: Create JSONL entries for each original file
    for audio_file in group_files:
        segment_id = extract_segment_id_from_filename(audio_file)
        
        # Get text from labels
        text = labels.get(segment_id, "").strip()
        if not text:
            print(f"Warning: No label found for {segment_id}, skipping", file=sys.stderr)
            continue
        
        # Get speaker info from mapping
        speaker_info = segment_mapping.get(segment_id, {})
        speaker_id = speaker_info.get("speaker")
        if not speaker_id:
            print(f"Warning: No speaker found for {segment_id}, skipping", file=sys.stderr)
            continue
        
        # Get duration from processed file
        processed_file = output_audio_dir / audio_file.name
        try:
            info = sf.info(str(processed_file))
            duration = info.duration
        except Exception:
            duration = 0.0
        
        # Create entry matching webui format
        # audio path format: "{output_root.name}/audio/{filename}"
        output_root_name = output_dir.name
        entry = {
            "id": segment_id,
            "text": text,
            "audio": f"{output_root_name}/audio/{audio_file.name}",
            "speaker": speaker_id,
            "language": "th",
            "duration": round(duration, 2),
            "source": audio_file.name,
        }
        entries.append(entry)
    
    return entries


def load_existing_jsonl_entries(jsonl_path: Path) -> set:
    """
    Load existing entry IDs from JSONL file to avoid duplicates.
    Returns a set of entry IDs.
    """
    existing_ids = set()
    if jsonl_path.exists():
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entry_id = entry.get("id")
                        if entry_id:
                            existing_ids.add(entry_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading existing JSONL: {e}", file=sys.stderr)
    return existing_ids


def generate_jsonl(entries: List[Dict[str, Any]], output_path: Path, append: bool = False):
    """
    Generate JSONL file with entries matching webui format.
    If append=True, append to existing file and skip duplicates.
    """
    existing_ids = set()
    if append and output_path.exists():
        existing_ids = load_existing_jsonl_entries(output_path)
    
    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        new_count = 0
        skipped_count = 0
        for entry in entries:
            entry_id = entry.get("id")
            if entry_id in existing_ids:
                skipped_count += 1
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            existing_ids.add(entry_id)
            new_count += 1
        
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} duplicate entries", file=sys.stderr)
        if new_count > 0:
            print(f"  Added {new_count} new entries", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Thai speaker labeling CLI tool"
    )
    parser.add_argument(
        "--tsv",
        type=str,
        required=True,
        help="Path to TSV file with filename and text labels"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Root directory containing audio files (will be searched recursively)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed files and JSONL"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Emilia config.json file"
    )
    parser.add_argument(
        "--dir-name",
        type=str,
        default="th",
        help="Directory name for speaker ID prefix (default: th)"
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.5,
        help="Silence duration between combined segments in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--uvr-workers",
        type=int,
        default=4,
        help="Number of parallel workers for UVR processing (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    tsv_path = Path(args.tsv)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    
    # Validate inputs
    if not tsv_path.exists():
        print(f"Error: TSV file not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)
    
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    output_audio_dir = output_dir / "audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    Logger.init_logger("th_speaker_labeling")
    logger = Logger.get_logger()
    
    # Load config
    config_path_resolved = config_path.resolve()
    cfg = load_cfg(str(config_path_resolved))
    
    # Resolve paths in config
    cfg["separate"]["step1"]["model_path"] = str(
        _resolve_path(config_path_resolved, cfg["separate"]["step1"]["model_path"])
    )
    if "mos_model" in cfg:
        cfg["mos_model"]["primary_model_path"] = str(
            _resolve_path(config_path_resolved, cfg["mos_model"]["primary_model_path"])
        )
    
    # Prepare models
    runtime_args = SimpleNamespace(
        batch_size=16,  # Not used for diarization, but required
        compute_type="float16" if args.device == "cuda" else "int8",
        whisper_arch="medium",  # Not used, but required
        threads=4,  # Not used, but required
        do_uvr=True  # Enable UVR
    )
    
    print("Initializing models...", file=sys.stderr)
    models = prepare_models(cfg, runtime_args)
    print("Models initialized.", file=sys.stderr)
    
    # Get target sample rate from config
    target_sample_rate = cfg.get("entrypoint", {}).get("SAMPLE_RATE", 24000)
    
    # Load TSV labels
    print(f"Loading labels from {tsv_path}...", file=sys.stderr)
    labels = load_tsv_labels(tsv_path)
    print(f"Loaded {len(labels)} labels.", file=sys.stderr)
    
    # Find and group audio files
    print(f"Finding audio files in {audio_dir}...", file=sys.stderr)
    audio_files = find_audio_files(audio_dir)
    print(f"Found {len(audio_files)} audio files.", file=sys.stderr)
    
    if not audio_files:
        print("Error: No audio files found!", file=sys.stderr)
        sys.exit(1)
    
    groups = group_audio_files(audio_files)
    print(f"Grouped into {len(groups)} groups.", file=sys.stderr)
    
    # Process each group
    all_entries = []
    for group_idx, (group_prefix, group_files) in enumerate(groups.items(), 1):
        print(f"\n[{group_idx}/{len(groups)}] Processing group: {group_prefix}", file=sys.stderr)
        entries = process_group(
            group_prefix,
            group_files,
            labels,
            models,
            output_dir,
            output_audio_dir,
            args.dir_name,
            args.silence_duration,
            target_sample_rate,
            args.uvr_workers
        )
        all_entries.extend(entries)
        print(f"Group {group_prefix}: Generated {len(entries)} entries.", file=sys.stderr)
    
    # Generate JSONL file (append mode to support resume)
    jsonl_path = output_dir / f"{args.dir_name}_transcribed.jsonl"
    print(f"\nGenerating JSONL file: {jsonl_path}", file=sys.stderr)
    generate_jsonl(all_entries, jsonl_path, append=True)
    
    print(f"\nComplete! Processed {len(all_entries)} entries in {jsonl_path}", file=sys.stderr)
    print(f"Processed audio files saved to: {output_audio_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
