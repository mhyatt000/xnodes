#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

from rich import print
import tyro


@dataclasses.dataclass
class Config:
    """
    Map specific /dev/video indices to stable by-path links:

      * look under /dev/v4l/by-path
      * only consider entries that end with 'video-index0'
      * resolve them to /dev/videoN
      * use low/side/over as those N's (e.g. 0, 2, 4)

    Then create:
      ~/.xnodes/dev/cam_low  -> /dev/v4l/by-path/<matching for videoN>
      ~/.xnodes/dev/cam_side -> ...
      ~/.xnodes/dev/cam_over -> ...
    """

    low: int  # e.g. 0
    side: int  # e.g. 2
    over: int  # e.g. 4

    dest_dir: Path = Path("~/.xnodes/dev").expanduser()
    loc: Path = Path("/dev/v4l/by-id")

    clear_existing: bool = True
    dry: bool = False


@dataclasses.dataclass
class CamEntry:
    video_idx: int  # N in /dev/videoN
    by_path: Path  # /dev/v4l/by-path/...
    target: Path  # /dev/videoN


def discover_by_video_index(loc: Path) -> dict[int, CamEntry]:
    """
    Return mapping: video_idx (N) -> CamEntry
    Only consider by-path entries whose name ends with 'video-index0',
    so we treat each physical camera once.
    """
    if not loc.is_dir():
        raise FileNotFoundError(f"{loc} does not exist or is not a directory")

    mapping: dict[int, CamEntry] = {}

    for p in sorted(loc.iterdir()):
        # Only take the "primary" stream for each camera
        if not p.name.endswith("video-index0"):
            continue
        print(p)
        try:
            target = p.resolve()
        except OSError:
            continue
        if not target.name.startswith("video"):
            continue
        try:
            vid_idx = int(target.name.removeprefix("video"))
        except ValueError:
            continue

        mapping[vid_idx] = CamEntry(video_idx=vid_idx, by_path=p, target=target)

    return mapping


def _select(mapping: dict[int, CamEntry], idx: int, label: str) -> CamEntry:
    if idx not in mapping:
        available = ", ".join(str(k) for k in sorted(mapping))
        raise KeyError(f"No by-path entry for /dev/video{idx} ({label}). Available primary video indices: {available}")
    return mapping[idx]


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mapping = discover_by_video_index(cfg.loc)
    print(mapping)
    if not mapping:
        raise RuntimeError(f"No primary by-path entries (*video-index0) under {cfg.loc}")

    logging.info("Discovered primary by-path entries (only *video-index0):")
    for vid_idx in sorted(mapping):
        entry = mapping[vid_idx]
        logging.info("  /dev/video%-2d  ->  %s", vid_idx, entry.by_path.name)

    low_entry = _select(mapping, cfg.low, "low")
    side_entry = _select(mapping, cfg.side, "side")
    over_entry = _select(mapping, cfg.over, "over")

    role_map = {
        "low": low_entry,
        "side": side_entry,
        "over": over_entry,
    }

    logging.info("")
    logging.info("Selected mapping:")
    for role, entry in role_map.items():
        logging.info(
            "  cam_%-4s -> /dev/video%-2d  -> by-path=%s",
            role,
            entry.video_idx,
            entry.by_path,
        )

    if cfg.dry:
        logging.info("Dry run enabled; not creating any symlinks.")
        return

    cfg.dest_dir.mkdir(parents=True, exist_ok=True)

    if cfg.clear_existing:
        for path in cfg.dest_dir.glob("cam_*"):
            if path.is_symlink():
                logging.info("Removing existing link %s", path)
                path.unlink()

    for role, entry in role_map.items():
        dest = cfg.dest_dir / f"cam_{role}"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        # Link to the by-path entry, not directly to /dev/videoN
        os.symlink(entry.by_path, dest)
        logging.info("Created %s -> %s (-> %s)", dest, entry.by_path, entry.target)


if __name__ == "__main__":
    main(tyro.cli(Config))
