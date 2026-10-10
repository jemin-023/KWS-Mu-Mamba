#!/usr/bin/env python3
"""
Download and extract the Google Speech Commands v0.02 dataset
(Hugging Face source: https://huggingface.co/datasets/google/speech_commands).

Structure:
  data/speech_commands/v0.02/
    ├── train/
    │   ├── backward/*.wav
    │   ├── bed/*.wav
    │   └── ...
    ├── validation/
    │   └── ...
    └── test/
        └── ...
"""

import os
import sys
import time
import tarfile
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DATASET_URLS = {
    "v0.02": {
        "test": "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.02/v0.02_test.tar.gz",
        "validation": "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.02/v0.02_validation.tar.gz",
        "train": "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.02/v0.02_train.tar.gz",
    },
    "v0.01": {
        "test": "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.01/v0.01_test.tar.gz",
        "validation": "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.01/v0.01_validation.tar.gz",
        "train": "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/v0.01/v0.01_train.tar.gz",
    },
}


def get_remote_file_size(url: str) -> int:
    """Fetch total size in bytes using a HEAD request."""
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "SpeechCommandsDownloader/1.0")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download_chunk(url: str, filepath: Path, start: int, end: int, max_retries: int = 5):
    """Download a byte range [start, end] and write directly to file at start offset."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={start}-{end}")
            req.add_header("User-Agent", "SpeechCommandsDownloader/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                expected_len = end - start + 1
                if len(data) != expected_len:
                    raise IOError(f"Incomplete chunk: expected {expected_len}, got {len(data)}")
                with open(filepath, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return len(data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed chunk {start}-{end} after {max_retries} attempts: {e}")
            time.sleep(1 + attempt * 2)


def download_file_parallel(url: str, dest_path: Path, num_workers: int = 16, chunk_size: int = 4 * 1024 * 1024):
    """Download a file using parallel range requests and display progress with tqdm."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total_size = get_remote_file_size(url)
    if total_size <= 0:
        raise ValueError(f"Could not retrieve content length for {url}")

    print(f"Downloading {dest_path.name} ({total_size / (1024 * 1024):.2f} MB) using {num_workers} parallel connections...")

    # Pre-allocate file size
    with open(dest_path, "wb") as f:
        f.truncate(total_size)

    # Determine chunks
    chunks = []
    start = 0
    while start < total_size:
        end = min(start + chunk_size - 1, total_size - 1)
        chunks.append((start, end))
        start = end + 1

    with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=dest_path.name) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {
                executor.submit(download_chunk, url, dest_path, s, e): (s, e) for s, e in chunks
            }
            for future in as_completed(future_to_chunk):
                bytes_downloaded = future.result()
                pbar.update(bytes_downloaded)

    actual_size = dest_path.stat().st_size
    if actual_size != total_size:
        raise ValueError(f"Downloaded file size ({actual_size}) != expected ({total_size})")


def extract_archive(tar_path: Path, extract_dir: Path):
    """Extract a tar.gz archive into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path.name} to {extract_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), unit="files", desc=f"Extracting {tar_path.name}") as pbar:
            for member in members:
                tar.extract(member, path=extract_dir)
                pbar.update(1)


def count_dataset_stats(base_dir: Path):
    """Print file and class statistics for the dataset directory."""
    print("\nDataset Summary:")
    print("=" * 60)
    splits = [d for d in sorted(base_dir.iterdir()) if d.is_dir()]
    total_samples = 0
    all_classes = set()

    for split in splits:
        classes = [c for c in sorted(split.iterdir()) if c.is_dir()]
        split_samples = 0
        for c in classes:
            wavs = list(c.glob("*.wav"))
            split_samples += len(wavs)
            all_classes.add(c.name)
        total_samples += split_samples
        print(f"  Split '{split.name}': {split_samples:,} wav files across {len(classes)} classes")

    print("=" * 60)
    print(f"Total Unique Classes: {len(all_classes)}")
    print(f"Total Samples Across All Splits: {total_samples:,}")
    print(f"Location: {base_dir.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Download and extract Google Speech Commands dataset.")
    parser.add_argument("--version", choices=["v0.01", "v0.02"], default="v0.02", help="Dataset version")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["test", "validation", "train"], help="Splits to download")
    parser.add_argument("--output-dir", type=str, default="data/speech_commands", help="Base directory to store dataset")
    parser.add_argument("--workers", type=int, default=16, help="Parallel download workers")
    parser.add_argument("--keep-tar", action="store_true", help="Keep .tar.gz archives after extraction")
    args = parser.parse_args()

    base_dir = Path(args.output_dir) / args.version
    base_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = base_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    urls = DATASET_URLS[args.version]

    for split in args.splits:
        url = urls[split]
        tar_name = f"{args.version}_{split}.tar.gz"
        tar_path = archive_dir / tar_name
        split_dir = base_dir / split

        print(f"\n--- Processing split: {split} ({args.version}) ---")

        # Download if tar doesn't exist
        if not tar_path.exists() or tar_path.stat().st_size == 0:
            download_file_parallel(url, tar_path, num_workers=args.workers)
        else:
            print(f"Archive {tar_path.name} already exists. Skipping download.")

        # Extract
        extract_archive(tar_path, split_dir)

        # Cleanup if not keeping
        if not args.keep_tar:
            print(f"Removing archive {tar_path.name} to free disk space...")
            tar_path.unlink()

    # Clean archive_dir if empty
    if not args.keep_tar and archive_dir.exists() and not any(archive_dir.iterdir()):
        archive_dir.rmdir()

    # Verify and summarize
    count_dataset_stats(base_dir)
    print("Download and extraction completed successfully!")


if __name__ == "__main__":
    main()
