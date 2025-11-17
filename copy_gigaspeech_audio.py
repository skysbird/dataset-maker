#!/usr/bin/env python3
"""
Copy GigaSpeech audio files based on TSV file.

Usage:
    python copy_gigaspeech_audio.py \
        --tsv-file /data/sky/gigaspeech/g2_th_refined/data/th/train_refined.tsv \
        --source-dir /data/sky/gigaspeech/g2_th_refined/data/th \
        --output-dir /path/to/output \
        [--split train]
"""

import argparse
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    print("Warning: soundfile not available, duration calculation will be skipped", file=sys.stderr)
    sf = None


def load_segment_ids(tsv_path: Path) -> list[str]:
    """
    Load segment IDs from TSV file.
    
    Format: segment_id\ttext
    Example: 0-0-0\tแต่ทีนี้ผู้สื่อข่าว...
    """
    segment_ids = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) >= 1:
                segment_id = parts[0].strip()
                if segment_id:
                    segment_ids.append(segment_id)
            else:
                print(f"Warning: Invalid line format at line {line_num}: {line[:50]}...", file=sys.stderr)
    return segment_ids


def get_audio_path(source_dir: Path, split: str, segment_id: str) -> Path:
    """
    Get audio file path from segment_id.
    
    segment_id format: 0-0-0
    Path format: train/0/0/0-0-0.wav
    """
    parts = segment_id.split("-")
    if len(parts) < 2:
        raise ValueError(f"Invalid segment_id format: {segment_id}")
    
    # First two parts form the directory path
    first_part = parts[0]
    second_part = parts[1]
    
    audio_path = source_dir / split / first_part / second_part / f"{segment_id}.wav"
    return audio_path


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    if sf is None:
        return 0.0
    try:
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        return 0.0


def copy_audio_files(
    tsv_file: Path,
    source_dir: Path,
    output_dir: Path,
    split: str = "train",
    preserve_structure: bool = False,
) -> None:
    """
    Copy audio files based on TSV file.
    
    Args:
        tsv_file: Path to TSV file with segment IDs
        source_dir: Root directory of GigaSpeech data (e.g., g2_th_refined/data/th)
        output_dir: Output directory for copied files
        split: Dataset split (train, dev, test)
        preserve_structure: If True, preserve directory structure; if False, flatten to output_dir
    """
    print(f"Loading segment IDs from {tsv_file}...")
    segment_ids = load_segment_ids(tsv_file)
    print(f"Found {len(segment_ids)} segment IDs")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    copied_count = 0
    missing_count = 0
    error_count = 0
    total_duration = 0.0
    
    print(f"\nCopying audio files to {output_dir}...")
    print(f"Source directory: {source_dir / split}")
    print(f"Preserve structure: {preserve_structure}\n")
    
    with tqdm(total=len(segment_ids), desc="Copying files") as pbar:
        for segment_id in segment_ids:
            try:
                # Get source path
                source_path = get_audio_path(source_dir, split, segment_id)
                
                if not source_path.exists():
                    missing_count += 1
                    pbar.set_postfix({
                        "copied": copied_count,
                        "missing": missing_count,
                        "errors": error_count,
                        "duration": f"{total_duration/3600:.1f}h"
                    })
                    pbar.update(1)
                    continue
                
                # Get duration
                duration = get_duration(source_path)
                total_duration += duration
                
                # Determine destination path
                if preserve_structure:
                    # Preserve structure: output_dir/train/0/0/0-0-0.wav
                    parts = segment_id.split("-")
                    first_part = parts[0]
                    second_part = parts[1]
                    dest_path = output_dir / split / first_part / second_part / f"{segment_id}.wav"
                else:
                    # Flatten: output_dir/0-0-0.wav
                    dest_path = output_dir / f"{segment_id}.wav"
                
                # Create parent directory if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                
                pbar.set_postfix({
                    "copied": copied_count,
                    "missing": missing_count,
                    "errors": error_count,
                    "duration": f"{total_duration/3600:.1f}h"
                })
                
            except Exception as e:
                error_count += 1
                print(f"\nError processing {segment_id}: {e}", file=sys.stderr)
                pbar.set_postfix({
                    "copied": copied_count,
                    "missing": missing_count,
                    "errors": error_count,
                    "duration": f"{total_duration/3600:.1f}h"
                })
            
            pbar.update(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Copy Summary:")
    print(f"  Total segments: {len(segment_ids)}")
    print(f"  Successfully copied: {copied_count}")
    print(f"  Missing files: {missing_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy GigaSpeech audio files based on TSV file"
    )
    parser.add_argument(
        "--tsv-file",
        type=Path,
        required=True,
        help="Path to TSV file with segment IDs (e.g., train_refined.tsv)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Root directory of GigaSpeech data (e.g., /data/sky/gigaspeech/g2_th_refined/data/th)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for copied audio files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve directory structure in output (default: flatten to single directory)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.tsv_file.exists():
        parser.error(f"TSV file not found: {args.tsv_file}")
    
    if not args.source_dir.exists():
        parser.error(f"Source directory not found: {args.source_dir}")
    
    train_dir = args.source_dir / args.split
    if not train_dir.exists():
        parser.error(f"Split directory not found: {train_dir}")
    
    # Copy files
    copy_audio_files(
        tsv_file=args.tsv_file,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split=args.split,
        preserve_structure=args.preserve_structure,
    )


if __name__ == "__main__":
    main()

