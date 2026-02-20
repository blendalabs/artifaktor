#!/usr/bin/env python3
"""Extract PNG frames from video files using ffmpeg.

Features:
- Input can be a single video file or a directory of videos.
- Configurable FPS (default: 3.0).
- Output PNGs with zero-padded filenames.
- Frames organized into one subdirectory per source video.

Example:
    python scripts/extract_frames.py sequences/my_video.mp4
    python scripts/extract_frames.py /data/videos --output-dir frames --fps 2.5
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass
class ExtractionResult:
    source: Path
    output_dir: Path
    frame_count: int
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PNG frames from videos using ffmpeg.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a video file or a directory containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frames"),
        help="Root directory where per-video frame folders will be created (default: ./frames).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Frame extraction rate in frames per second (default: 3.0).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated list of video extensions for directory mode.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive directory scan when input_path is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if output directory already contains PNG files.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable name/path (default: ffmpeg).",
    )
    return parser.parse_args()


def normalize_extensions(raw: str) -> set[str]:
    exts = set()
    for item in raw.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.add(ext)
    return exts


def find_videos(input_path: Path, extensions: set[str], recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    glob_pattern = "**/*" if recursive else "*"
    files = [p for p in input_path.glob(glob_pattern) if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files)


def ensure_ffmpeg(ffmpeg_bin: str) -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(
            f"Could not find ffmpeg executable: '{ffmpeg_bin}'. Install ffmpeg or pass --ffmpeg-bin."
        )


def safe_subdir_name(video_path: Path, output_root: Path) -> Path:
    """Create a stable output subdir based on source filename.

    If a collision occurs, append a numeric suffix.
    """
    base = video_path.stem
    candidate = output_root / base
    if not candidate.exists():
        return candidate

    # If it exists, keep first one if it's the same source stem usage pattern;
    # otherwise find a unique directory name.
    idx = 2
    while True:
        numbered = output_root / f"{base}_{idx}"
        if not numbered.exists():
            return numbered
        idx += 1


def count_pngs(path: Path) -> int:
    return len([p for p in path.glob("*.png") if p.is_file()])


def run_ffmpeg(
    ffmpeg_bin: str,
    source_video: Path,
    output_dir: Path,
    fps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / f"{output_dir.name}.%06d.png"

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y",
        "-i",
        str(source_video),
        "-vf",
        f"fps={fps}",
        "-start_number",
        "1",
        str(output_pattern),
    ]

    subprocess.run(cmd, check=True)


def extract_many(
    ffmpeg_bin: str,
    videos: Iterable[Path],
    output_root: Path,
    fps: float,
    overwrite: bool,
) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for video in videos:
        if not video.exists():
            print(f"[WARN] Skipping missing file: {video}")
            continue

        target_dir = safe_subdir_name(video, output_root)

        if target_dir.exists() and count_pngs(target_dir) > 0 and not overwrite:
            frame_count = count_pngs(target_dir)
            print(f"[SKIP] {video} -> {target_dir} ({frame_count} existing frames)")
            results.append(
                ExtractionResult(source=video, output_dir=target_dir, frame_count=frame_count, skipped=True)
            )
            continue

        print(f"[RUN ] {video} -> {target_dir}")
        run_ffmpeg(ffmpeg_bin=ffmpeg_bin, source_video=video, output_dir=target_dir, fps=fps)
        frame_count = count_pngs(target_dir)
        print(f"[DONE] {target_dir} ({frame_count} frames)")
        results.append(ExtractionResult(source=video, output_dir=target_dir, frame_count=frame_count))

    return results


def main() -> int:
    args = parse_args()

    input_path: Path = args.input_path.resolve()
    output_dir: Path = args.output_dir.resolve()
    fps: float = args.fps

    if fps <= 0:
        print("[ERROR] --fps must be > 0", file=sys.stderr)
        return 2

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        return 2

    extensions = normalize_extensions(args.extensions)
    recursive = not args.no_recursive

    try:
        ensure_ffmpeg(args.ffmpeg_bin)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    videos = find_videos(input_path=input_path, extensions=extensions, recursive=recursive)
    if not videos:
        print(
            "[ERROR] No video files found. "
            f"Input={input_path} Extensions={sorted(extensions)} Recursive={recursive}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(videos)} video(s). Output root: {output_dir}. FPS: {fps}")

    try:
        results = extract_many(
            ffmpeg_bin=args.ffmpeg_bin,
            videos=videos,
            output_root=output_dir,
            fps=fps,
            overwrite=args.overwrite,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    generated = sum(r.frame_count for r in results if not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    print(f"\nSummary: processed={len(results)} skipped={skipped} generated_frames={generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
