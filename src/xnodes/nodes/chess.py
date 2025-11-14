from dataclasses import dataclass, field
from rich import print
from geometry_msgs.msg import TransformStamped
import os.path as osp
from tf2_ros import TransformBroadcaster

import numpy as np
from typing import List, Tuple

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray

import transforms3d.quaternions as tq
import tyro


@dataclass
class ChessConfig:
    sub: str  # image topic
    board: List[int] = field(
        default_factory=lambda: [8, 6]
    )  # inner corners: [cols, rows]
    square_size: float = (
        1.0  # meters per square edge; set correctly for metric extrinsics
    )
    min_samples: int = 15  # calibrate after this many good detections
    show: bool = False  # draw corners in a window
    use_sb: bool = True  # use findChessboardCornersSB if True

    @property
    def square_size_in2m(self) -> float:
        return self.square_size * 0.0254  # inches to meters


def arr2poses(arr: np.ndarray) -> PoseArray:
    poses = PoseArray()
    poses.poses = []
    for i in range(arr.shape[0]):
        p = Pose()
        p.position.x = float(arr[i, 0])
        p.position.y = float(arr[i, 1])
        p.position.z = float(arr[i, 2])
        # Orientation left as default (0,0,0,1)
        poses.poses.append(p)
    return poses


def mat4_to_transform(T: np.ndarray) -> TransformStamped:
    # T is 4x4 homogeneous matrix
    msg = TransformStamped()

    # translation comes from the last column
    msg.transform.translation.x = float(T[0, 3])
    msg.transform.translation.y = float(T[1, 3])
    msg.transform.translation.z = float(T[2, 3])

    # rotation: convert 3x3 to quaternion
    q = tq.mat2quat(T[:3, :3])  # returns [w, x, y, z]

    msg.transform.rotation.w = float(q[0])
    msg.transform.rotation.x = float(q[1])
    msg.transform.rotation.y = float(q[2])
    msg.transform.rotation.z = float(q[3])

    return msg


class Chess(Node):
    def __init__(self, cfg: ChessConfig):
        super().__init__("chess_calibrator")
        self.cfg = cfg
        self.bridge = CvBridge()

        # OpenCV expects inner-corner grid size: (cols, rows)
        self.pattern_size: Tuple[int, int] = (cfg.board[0], cfg.board[1])
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1e-3,
        )

        # Pre-build the single board’s 3D points in the board frame
        # z = 0 plane; scale by square size to get metric extrinsics
        objp = np.zeros((1, self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[
            0 : self.pattern_size[0], 0 : self.pattern_size[1]
        ].T.reshape(-1, 2)
        objp *= float(cfg.square_size_in2m)

        self._template_objp = objp
        self.objpoints = []  # list of (N,1,3)
        self.imgpoints = []  # list of (N,1,2)
        self.image_size = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # most camera drivers use this
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub = self.create_subscription(Image, cfg.sub, self._on_image, qos)
        self.pub = self.create_publisher(PoseArray, osp.join(cfg.sub[1:], "chess"), 10)
        self.bst = TransformBroadcaster(self)
        self.parent = None

        self.get_logger().info(
            f"Chess listening on {cfg.sub} for {self.pattern_size} inner corners"
        )

    def _find_corners(self, gray):
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
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

    def _on_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        if self.image_size is None:
            self.image_size = (cv_img.shape[1], cv_img.shape[0])  # (w, h)
        self.parent = msg.header.frame_id  # to publish the points tf

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        ok, corners = self._find_corners(gray)

        if ok:
            self.objpoints.append(self._template_objp.copy())
            self.imgpoints.append(corners.astype(np.float32))

            if self.cfg.show:
                vis = cv_img.copy()
                cv2.drawChessboardCorners(vis, self.pattern_size, corners, ok)
                cv2.imshow("chess", vis)
                cv2.waitKey(1)

            n = len(self.imgpoints)
            if n % 5 == 0 or n == self.cfg.min_samples:
                self.get_logger().info(f"collected {n} good views")

            if len(self.imgpoints) >= self.cfg.min_samples:
                self._try_calibrate()

        else:
            if self.cfg.show:
                cv2.imshow("chess", cv_img)
                cv2.waitKey(1)

        # print(np.array(self.objpoints).shape)
        # print(np.array(self.imgpoints).shape)

    def _try_calibrate(self):
        if self.image_size is None or not self.imgpoints:
            return
        self.get_logger().info("running calibrateCamera...")
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
            R, _ = cv2.Rodrigues(rvecs[0])
            t = tvecs[0].reshape(3)
            self.get_logger().info(
                f"first view extrinsic:\nR=\n{R}\nt(meters)={t}  "
                f"(scale depends on square size={self.cfg.square_size_in2m})"
            )

        # Optional: save to YAML
        fs = cv2.FileStorage("camera_calib.yaml", cv2.FILE_STORAGE_WRITE)
        fs.write("K", K)
        fs.write("dist", dist)
        fs.write("image_width", int(self.image_size[0]))
        fs.write("image_height", int(self.image_size[1]))
        fs.release()
        self.get_logger().info("wrote camera_calib.yaml")

        objp = self._template_objp.reshape(-1, 3).T  # (3, N)
        X_cam = ((R @ objp) + t.reshape(-1, 1)).T  # (3, N)

        # print(X_cam)

        poses = arr2poses(X_cam)
        self.pub.publish(poses)

        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = t.reshape(3)

        t = mat4_to_transform(mat)
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent  # parent frame
        prefix = self.parent.split("_")[0]
        t.child_frame_id = f"{prefix}_chessboard"  # child frame
        print(t)

        self.bst.sendTransform(t)

        # After calibration, stop accumulating to avoid re-fitting on the same dataset
        # self.objpoints.clear()
        # self.imgpoints.clear()
        self.objpoints, self.imgpoints = (
            self.objpoints[-self.cfg.min_samples :],
            self.imgpoints[-self.cfg.min_samples :],
        )

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
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
