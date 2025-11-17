#!/usr/bin/env python3
"""
Apply UVR (Ultimate Vocal Remover) to GigaSpeech wav files in-place.

This script processes wav files listed in the JSONL output from process_gigaspeech.py,
applies UVR to remove background music, and saves the processed audio
back to the original location (overwriting or saving alongside).

Usage:
    python uvr_gigaspeech.py \
        --gigaspeech-root g2_th_refined/data/th \
        --jsonl output/gigaspeech_processed/train_with_speakers.jsonl \
        --config Emilia/config.json \
        --backup  # Optional: backup original files
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from infer_uvr import UVRSeparator
except ImportError:
    print("Error: Could not import UVRSeparator from infer_uvr", file=sys.stderr)
    sys.exit(1)


def load_audio_files_from_jsonl(
    jsonl_path: Path,
    gigaspeech_root: Path,
) -> List[Path]:
    """
    Load audio file paths from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file (e.g., train_with_speakers.jsonl)
        gigaspeech_root: Root directory of GigaSpeech data
    
    Returns:
        List of audio file paths (absolute paths)
    """
    audio_files = []
    seen_paths: Set[str] = set()
    
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found: {jsonl_path}", file=sys.stderr)
        return audio_files
    
    print(f"Loading audio files from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                audio_rel_path = entry.get("audio")
                if not audio_rel_path:
                    continue
                
                # Build absolute path
                audio_path = gigaspeech_root / audio_rel_path
                
                # Deduplicate (same file might appear multiple times)
                audio_path_str = str(audio_path.resolve())
                if audio_path_str in seen_paths:
                    continue
                seen_paths.add(audio_path_str)
                
                if audio_path.exists():
                    audio_files.append(audio_path)
                else:
                    print(f"Warning: Audio file not found: {audio_path} (line {line_num})", file=sys.stderr)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue
    
    return sorted(audio_files)


def process_audio_file(
    audio_path: Path,
    separator: UVRSeparator,
    backup: bool = False,
) -> bool:
    """
    Process a single audio file with UVR and save in-place.
    
    Args:
        audio_path: Path to the audio file
        separator: UVRSeparator instance
        backup: If True, create backup before overwriting
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load original audio
        waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        
        # Convert to stereo if needed
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform])
        if waveform.shape[0] > 2:
            waveform = waveform[:2]
        
        # Run UVR separation
        background, vocals = separator.predict(waveform, sample_rate)
        
        # Use vocals only (convert to mono)
        if vocals.ndim == 1:
            vocal_mono = vocals
        elif vocals.ndim == 2:
            # Take mean across channels
            channel_axis = 0 if vocals.shape[0] < vocals.shape[1] else 1
            vocal_mono = np.mean(vocals, axis=channel_axis, keepdims=False)
        else:
            vocal_mono = np.mean(vocals, axis=tuple(range(vocals.ndim - 1)))
        
        # Ensure sample rate matches original
        if sample_rate != 24000:
            vocal_mono = librosa.resample(vocal_mono, orig_sr=sample_rate, target_sr=24000)
            sample_rate = 24000
        
        # Backup original file if requested
        if backup:
            backup_path = audio_path.with_suffix('.wav.backup')
            if not backup_path.exists():
                shutil.copy2(audio_path, backup_path)
        
        # Save processed audio (overwrite original)
        sf.write(str(audio_path), vocal_mono, sample_rate)
        
        return True
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply UVR to GigaSpeech wav files listed in JSONL file"
    )
    parser.add_argument(
        "--gigaspeech-root",
        type=Path,
        required=True,
        help="Root directory of GigaSpeech data (e.g., g2_th_refined/data/th)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to JSONL file with audio paths (e.g., output/gigaspeech_processed/train_with_speakers.jsonl)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Emilia/config.json"),
        help="Path to Emilia config.json (default: Emilia/config.json)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original files before processing",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.gigaspeech_root.exists():
        parser.error(f"GigaSpeech root directory not found: {args.gigaspeech_root}")
    
    if not args.jsonl.exists():
        parser.error(f"JSONL file not found: {args.jsonl}")
    
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")
    
    # Load config
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        parser.error(f"Failed to load config file: {e}")
    
    # Get UVR model path (using same resolution logic as emilia_pipeline.py)
    uvr_model_path_str = cfg["separate"]["step1"]["model_path"]
    uvr_model_path = Path(uvr_model_path_str)
    
    if not uvr_model_path.is_absolute():
        # Try relative to config file first
        relative_to_config = args.config.parent / uvr_model_path
        if relative_to_config.exists():
            uvr_model_path = relative_to_config.resolve()
        else:
            # Try relative to current working directory
            relative_to_cwd = Path.cwd() / uvr_model_path
            if relative_to_cwd.exists():
                uvr_model_path = relative_to_cwd.resolve()
            else:
                # Fall back to config-relative (may not exist yet)
                uvr_model_path = relative_to_config.resolve(strict=False)
    
    if not uvr_model_path.exists():
        parser.error(f"UVR model not found: {uvr_model_path}\n"
                     f"  Tried: {args.config.parent / uvr_model_path_str}\n"
                     f"  Tried: {Path.cwd() / uvr_model_path_str}")
    
    # Initialize UVR separator
    print(f"Initializing UVR separator with model: {uvr_model_path}")
    try:
        separator = UVRSeparator(
            uvr_model_path,
            metadata_json=cfg["separate"]["step1"].get("metadata_json"),
        )
        print(f"UVR separator initialized successfully")
    except Exception as e:
        parser.error(f"Failed to initialize UVR separator: {e}")
    
    # Load audio files from JSONL
    wav_files = load_audio_files_from_jsonl(args.jsonl, args.gigaspeech_root)
    
    if not wav_files:
        print("No audio files found in JSONL!")
        return
    
    if args.max_files:
        wav_files = wav_files[:args.max_files]
        print(f"Limiting to {args.max_files} files for testing")
    
    print(f"Found {len(wav_files)} unique audio files to process")
    
    if args.backup:
        print("Backup mode enabled: original files will be saved as .wav.backup")
    
    # Process files
    success_count = 0
    fail_count = 0
    processed_files = []
    failed_files = []
    
    for audio_path in tqdm(wav_files, desc="Processing audio files"):
        if process_audio_file(audio_path, separator, backup=args.backup):
            success_count += 1
            processed_files.append(audio_path)
        else:
            fail_count += 1
            failed_files.append(audio_path)
    
    print(f"\nProcessing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(wav_files)}")
    
    # Print processed files
    if processed_files:
        print(f"\n✓ Successfully processed {len(processed_files)} files:")
        for audio_path in processed_files:
            print(f"  {audio_path}")
    
    # Print failed files if any
    if failed_files:
        print(f"\n✗ Failed to process {len(failed_files)} files:")
        for audio_path in failed_files:
            print(f"  {audio_path}")


if __name__ == "__main__":
    main()

