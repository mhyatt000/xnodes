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


def expand_points(points, scale=1.25):
    """
    points: (N, 2) array-like
    scale: factor to move points away from center
    returns: (N, 2) scaled points
    """
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)  # compute centroid
    expanded = center + (pts - center) * scale
    return expanded


def _corners_3d_from_obj(row: int, col: int, cfg: AprilGridConfig) -> np.ndarray:
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


def _corners_3d_from_pose(pose_R, pose_t, tag_size):
    """
    Compute the AprilTag's 4 corners in the camera frame.

    pose_R: (3,3) rotation matrix (tag->camera)
    pose_t: (3,1) translation vector (tag->camera)
    tag_size: side length of the tag in meters
    """
    half = tag_size / 2.0

    # Tag-frame corners (origin at tag center, z=0)
    corners_tag = np.array(
        [
            [-half, half, 0.0],  # top-left
            [half, half, 0.0],  # top-right
            [half, -half, 0.0],  # bottom-right
            [-half, -half, 0.0],  # bottom-left
        ]
    )  # (4,3)

    # Transform to camera frame: R @ p + t
    corners_tag_T = corners_tag.T  # (3,4)
    corners_cam_T = pose_R @ corners_tag_T + pose_t  # (3,4)
    return corners_cam_T.T  # (4,3)


def project_points(K, pts_3d):
    """
    Project 3D camera-frame points to 2D image.

    K: (3,3) camera intrinsics
    pts_3d: (N,3) points in camera frame
    """
    pts = pts_3d.T  # (3,N)

    # Normalize by Z
    pts_norm = pts / pts[2]

    # Project
    pix_h = K @ pts_norm  # (3,N)
    pix = (pix_h[:2] / pix_h[2]).T  # (N,2)

    return pix


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
    camera_params: list[float] | None = None,
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
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=cfg.tagSize,
    )

    all_pts2d = []
    all_pts3d = []
    tag_obs = []

    if detections:
        print(detections[0])
    else:
        return

    for det in detections:
        tag_id = det.tag_id
        rc = _tag_row_col(tag_id, cfg)
        if rc is None:
            continue  # tag not part of this board

        row, col = rc

        # corners: (4, 2), order: TL, TR, BR, BL
        corners2d = np.asarray(det.corners, dtype=np.float32)  # (4, 2)
        corners3d = _corners_3d_from_obj(row, col, cfg)  # (4, 3)

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
