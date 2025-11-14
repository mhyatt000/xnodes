"""Core calibration metrics utilities."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def corner_grid_rms(corners: NDArray[np.float32], pattern_size: tuple[int, int]) -> float:
    """Compute RMS reprojection error for chessboard corners."""
    cols, rows = pattern_size
    ideal = np.stack(
        np.meshgrid(
            np.arange(cols, dtype=np.float32),
            np.arange(rows, dtype=np.float32),
        ),
        axis=-1,
    ).reshape(-1, 2)

    pts = corners.reshape(-1, 2).astype(np.float32)

    affine, _ = cv2.estimateAffine2D(ideal, pts, method=cv2.LMEDS)
    if affine is None:
        raise ValueError("Affine fit failed for the provided corners.")

    ideal_h = np.hstack([ideal, np.ones((len(ideal), 1), dtype=np.float32)])
    pred = ideal_h @ affine.T

    err = np.linalg.norm(pred - pts, axis=1)
    return float(np.sqrt(np.mean(err**2)))
