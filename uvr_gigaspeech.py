#!/usr/bin/env python3
"""
Apply UVR (Ultimate Vocal Remover) to GigaSpeech wav files in-place.

This script processes all wav files in the GigaSpeech directory structure,
applies UVR to remove background music, and saves the processed audio
back to the original location (overwriting or saving alongside).

Usage:
    python uvr_gigaspeech.py \
        --gigaspeech-root g2_th_refined/data/th \
        --config Emilia/config.json \
        --split train \
        --backup  # Optional: backup original files
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from infer_uvr import UVRSeparator
except ImportError:
    print("Error: Could not import UVRSeparator from infer_uvr", file=sys.stderr)
    sys.exit(1)


def find_all_wav_files(root_dir: Path, split: str = "train") -> List[Path]:
    """
    Recursively find all .wav files in GigaSpeech directory structure.
    
    Args:
        root_dir: Root directory (e.g., g2_th_refined/data/th)
        split: Dataset split (train, dev, test)
    
    Returns:
        List of all .wav file paths
    """
    wav_files = []
    split_dir = root_dir / split
    
    if not split_dir.exists():
        print(f"Error: Directory not found: {split_dir}", file=sys.stderr)
        return wav_files
    
    # Walk through directory structure: train/10/10000/*.wav
    for subdir1 in split_dir.iterdir():
        if not subdir1.is_dir():
            continue
        for subdir2 in subdir1.iterdir():
            if not subdir2.is_dir():
                continue
            # Collect all .wav files in this directory
            wav_files.extend(sorted(subdir2.glob("*.wav")))
    
    return wav_files


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
        description="Apply UVR to GigaSpeech wav files in-place"
    )
    parser.add_argument(
        "--gigaspeech-root",
        type=Path,
        required=True,
        help="Root directory of GigaSpeech data (e.g., g2_th_refined/data/th)",
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
    
    # Find all wav files
    print(f"Scanning for wav files in {args.gigaspeech_root / args.split}...")
    wav_files = find_all_wav_files(args.gigaspeech_root, args.split)
    
    if not wav_files:
        print("No wav files found!")
        return
    
    if args.max_files:
        wav_files = wav_files[:args.max_files]
        print(f"Limiting to {args.max_files} files for testing")
    
    print(f"Found {len(wav_files)} wav files to process")
    
    if args.backup:
        print("Backup mode enabled: original files will be saved as .wav.backup")
    
    # Process files
    success_count = 0
    fail_count = 0
    
    for audio_path in tqdm(wav_files, desc="Processing audio files"):
        if process_audio_file(audio_path, separator, backup=args.backup):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nProcessing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(wav_files)}")


if __name__ == "__main__":
    main()

