from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pupil_apriltags import Detector
from reader import AprilGridConfig
from rich import print


@dataclass
class TagObs:
    tag_id: int
    row: int
    col: int
    corners2d: np.ndarray  # (4, 2)
    corners3d: np.ndarray  # (4, 3)


def _tag_row_col(tag_id: int, cfg: AprilGridConfig) -> tuple[int, int] | None:
    """Map absolute tag id -> (row, col) in the board, or None if not on board."""
    idx = tag_id - cfg.codeOffset
    if idx < 0 or idx >= cfg.num_tags:
        return None
    r = idx // cfg.tagCols
    c = idx % cfg.tagCols
    return r, c


def _tag_corners_3d(row: int, col: int, cfg: AprilGridConfig) -> np.ndarray:
    """
    3D corners in board frame, shape (4, 3).

    Convention:
      - Board lies in z=0 plane.
      - x: right, y: down.
      - Origin at board center.
      - Corner order matches typical apriltag output:
        [top-left, top-right, bottom-right, bottom-left].
    """
    spacing = cfg.center_spacing
    # board center at (0, 0)
    cx = (col - (cfg.tagCols - 1) / 2.0) * spacing
    cy = (row - (cfg.tagRows - 1) / 2.0) * spacing

    half = cfg.tagSize / 2.0
    # top-left, top-right, bottom-right, bottom-left
    corners = np.array(
        [
            [cx - half, cy - half, 0.0],
            [cx + half, cy - half, 0.0],
            [cx + half, cy + half, 0.0],
            [cx - half, cy + half, 0.0],
        ],
        dtype=np.float32,
    )
    return corners


# --- detector factory ---


def make_apriltag_detector(
    families: str = "tag36h11",
    nthreads: int = 16,
    quad_decimate: float = 1.0,
    quad_sigma: float = 0.0,
    refine_edges: int = 1,
    decode_sharpening: float = 0.25,
    debug: int = 1,
) -> Detector:
    """
    Pupil-AprilTags detector.
    """
    return Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )


def detect_aprilgrid(
    image_gray: np.ndarray,
    cfg: AprilGridConfig,
    detector: Detector,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Detect AprilTag grid correspondences with pupil-apriltags.

    Args:
        image_gray: HxW uint8 grayscale image.
        cfg: AprilGridConfig loaded from YAML.
        detector: pupil_apriltags.Detector.

    Returns:
        pts2d: (4N, 2) image points
        pts3d: (4N, 3) board-frame points
        or None if no board tags found.
    """
    # pupil-apriltags API
    detections = detector.detect(
        image_gray,
        estimate_tag_pose=False,  # we do our own pose if needed
        camera_params=None,
        tag_size=cfg.tagSize,
    )

    print(detections)
    all_pts2d = []
    all_pts3d = []
    tag_obs = []

    for det in detections:
        tag_id = det.tag_id
        rc = _tag_row_col(tag_id, cfg)
        if rc is None:
            continue  # tag not part of this board

        row, col = rc

        # corners: (4, 2), order: TL, TR, BR, BL
        corners2d = np.asarray(det.corners, dtype=np.float32)  # (4, 2)
        corners3d = _tag_corners_3d(row, col, cfg)  # (4, 3)

        all_pts2d.append(corners2d)
        all_pts3d.append(corners3d)
        tag_obs.append(
            TagObs(
                tag_id=det.tag_id,
                row=row,
                col=col,
                corners2d=corners2d,
                corners3d=corners3d,
            )
        )

    if not all_pts2d:
        return None

    print(len(all_pts2d), "tags detected")
    pts2d = np.concatenate(all_pts2d, axis=0)  # (4N, 2)
    pts3d = np.concatenate(all_pts3d, axis=0)  # (4N, 3)
    return pts2d, pts3d, tag_obs, detections
