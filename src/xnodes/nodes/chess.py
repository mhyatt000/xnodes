from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rich import print
from sensor_msgs.msg import Image
import tyro

from xnodes.chess_util import arr2poses, mat2quat, timeit


@dataclass
class ChessConfig:
    sub: str  # image topic
    board: list[int] = field(default_factory=lambda: [9, 6])  # inner corners: [cols, rows]
    square_size: float = 22.5  # mm
    n: int = 15  # target number of views to keep
    show: bool = False  # draw corners in a window
    use_sb: bool = True  # use findChessboardCornersSB if True

    @property
    def square_size_in2m(self) -> float:
        return self.square_size * 0.0254  # inches to meters


def _pose_features(rvecs: list[np.ndarray], tvecs: list[np.ndarray]) -> np.ndarray:
    """
    Build feature matrix [x, y, z, qw, qx, qy, qz] for each view.
    Returns array of shape (m, 7).
    """
    m = len(rvecs)
    feats = np.zeros((m, 7), dtype=np.float64)
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rv)
        t = tv.reshape(3)
        q = mat2quat(R)
        feats[i, 0:3] = t
        feats[i, 3:7] = q
    return feats


def plot_pose_pca(rvecs, tvecs, title: str = "Pose PCA (PC1 vs PC2)"):
    """
    rvecs, tvecs: lists of (3,1) or (3,) arrays from calibrateCamera.

    Builds features [x, y, z, qw, qx, qy, qz], runs PCA, and plots PC1 vs PC2.
    """
    m = len(rvecs)
    if m == 0:
        print("No views to plot.")
        return

    feats = np.zeros((m, 7), dtype=np.float64)
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rv)
        t = tv.reshape(3)
        q = mat2quat(R)
        feats[i, 0:3] = t
        feats[i, 3:7] = q

    # Center and PCA via SVD
    Xc = feats - feats.mean(axis=0, keepdims=True)
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T  # (m, 7)

    x = Z[:, 0]
    y = Z[:, 1]

    plt.figure()
    plt.scatter(x, y)
    # annotate with indices for debugging / view id
    for i in range(m):
        plt.text(x[i], y[i], str(i), fontsize=8, ha="center", va="center")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.pause(0.1)
    plt.clf()


class Chess(Node):
    def __init__(self, cfg: ChessConfig):
        super().__init__("chess_calibrator")
        self.cfg = cfg
        self.bridge = CvBridge()

        # OpenCV expects inner-corner grid size: (cols, rows)
        self.pattern_size: tuple[int, int] = (cfg.board[0], cfg.board[1])
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1e-3,
        )

        # Pre-build the single board's 3D points in the board frame
        # z = 0 plane; scale by square size to get metric extrinsics
        objp = np.zeros((1, self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0 : self.pattern_size[0], 0 : self.pattern_size[1]].T.reshape(-1, 2)
        objp *= float(cfg.square_size_in2m)

        self._template_objp = objp
        self.objpoints: list[np.ndarray] = []  # list of (1,N,3)
        self.imgpoints: list[np.ndarray] = []  # list of (N,1,2)
        self.image_size: tuple[int, int] | None = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # most camera drivers use this
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub = self.create_subscription(Image, cfg.sub, self._on_image, qos)
        self.mypub = self.create_publisher(PoseArray, "chess", 10)

        self.get_logger().info(
            f"Chess listening on {cfg.sub} for {self.pattern_size} inner corners; targeting {self.cfg.n} views"
        )

    @timeit
    def _find_corners(self, gray):
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        if self.cfg.use_sb and hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, flags=0)
            if ok and corners is not None and corners.ndim == 2:
                # SB returns (N,2); calibrateCamera wants (N,1,2)
                corners = corners.reshape(-1, 1, 2).astype(np.float32)
            return ok, corners
        else:
            ok, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
            if ok:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=self.criteria,
                )
            return ok, corners

    @timeit
    def _on_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        if self.image_size is None:
            self.image_size = (cv_img.shape[1], cv_img.shape[0])  # (w, h)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        binary = (gray > 127).astype(np.uint8) * 255
        ok, corners = self._find_corners(gray)

        if ok:
            self.objpoints.append(self._template_objp.copy())
            self.imgpoints.append(corners.astype(np.float32))

            n = len(self.imgpoints)
            if n % 5 == 0 or n == self.cfg.n:
                self.get_logger().info(f"collected {n} good views (keeping up to {self.cfg.n})")

            if len(self.imgpoints) >= self.cfg.n:
                print(len(self.imgpoints), self.cfg.n)
                self._try_calibrate()

        if self.cfg.show:
            vis = cv_img.copy()
            cv2.drawChessboardCorners(vis, self.pattern_size, corners, ok)
            cv2.imshow("chess", vis)
            cv2.waitKey(1)

    @timeit
    def _select_views_by_spread(
        self,
        rvecs: list[np.ndarray],
        tvecs: list[np.ndarray],
    ) -> list[int]:
        """
        Greedy farthest-point sampling on pose features [x, y, z, qw, qx, qy, qz].

        Returns indices of views to keep (at most cfg.n) that are maximally spread out
        in this feature space.
        """
        m = len(rvecs)
        if m <= self.cfg.n:
            return list(range(m))

        feats = _pose_features(rvecs, tvecs)  # (m, 7)

        # Standardize each dimension so translation and rotation have comparable weight
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0  # avoid division by zero
        X = (feats - mean) / std  # (m, 7)

        n_keep = self.cfg.n
        selected: list[int] = []

        # Start with point of maximum norm
        norms = np.sum(X**2, axis=1)
        first = int(np.argmax(norms))
        selected.append(first)

        # min distance to current selected set
        min_d = np.full(m, np.inf)
        while len(selected) < n_keep:
            last = selected[-1]
            d = np.linalg.norm(X - X[last], axis=1)
            min_d = np.minimum(min_d, d)
            # Never select already selected indices
            min_d[selected] = -np.inf
            nxt = int(np.argmax(min_d))
            selected.append(nxt)

        selected.sort()
        return selected

    @timeit
    def _try_calibrate(self):
        if self.image_size is None or not self.imgpoints:
            return

        self.get_logger().info(f"running calibrateCamera with {len(self.imgpoints)} views...")
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=self.objpoints,
            imagePoints=self.imgpoints,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        self.get_logger().info(f"rms reprojection error: {ret:.4f}")
        self.get_logger().info(f"K:\n{K}\ndist:\n{dist.squeeze()}")

        # Report one example extrinsic with metric scale if square size > 0
        if rvecs and tvecs:
            R0, _ = cv2.Rodrigues(rvecs[0])
            t0 = tvecs[0].reshape(3)
            self.get_logger().info(
                f"first view extrinsic:\nR=\n{R0}\nt(meters)={t0}  "
                f"(scale depends on square size={self.cfg.square_size_in2m})"
            )

        print(K)

        # Optional: save to YAML
        fs = cv2.FileStorage("camera_calib.yaml", cv2.FILE_STORAGE_WRITE)
        fs.write("K", K)
        fs.write("dist", dist)
        fs.write("image_width", int(self.image_size[0]))
        fs.write("image_height", int(self.image_size[1]))
        fs.release()
        self.get_logger().info("wrote camera_calib.yaml")

        # Example board points in camera frame for first view
        if rvecs and tvecs:
            objp = self._template_objp.reshape(-1, 3).T  # (3, N)
            X_cam = ((R0 @ objp) + t0.reshape(-1, 1)).T  # (N, 3)
            print(X_cam.shape)

            poses = arr2poses(X_cam)
            self.mypub.publish(poses)

        # Now, run PCA selection on all views and keep only the most "spread" ones
        if rvecs and tvecs:
            plot_pose_pca(rvecs, tvecs, title="Before PCA Selection")
            keep_idx = self._select_views_by_spread(rvecs, tvecs)
            old_n = len(self.imgpoints)
            self.objpoints = [self.objpoints[i] for i in keep_idx]
            self.imgpoints = [self.imgpoints[i] for i in keep_idx]
            self.get_logger().info(f"PCA selection: reduced views from {old_n} -> {len(self.imgpoints)}")

    def destroy_node(self):
        with contextlib.suppress(Exception):
            cv2.destroyAllWindows()
        super().destroy_node()


def main():
    cfg = tyro.cli(ChessConfig)
    rclpy.init()
    node = Chess(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Try to calibrate with whatever we have before shutdown
        node._try_calibrate()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
