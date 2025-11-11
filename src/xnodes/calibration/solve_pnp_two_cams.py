# solve_pnp_two_cams.py
import json, cv2, numpy as np
from pathlib import Path

# === Paths to the simultaneous checkerboard photos you saved ===
IMG_CAM0 = "checkerboard_captures/cam0_00_1762205304731.jpg"  # side cam
IMG_CAM2 = "checkerboard_captures/cam2_00_1762205304731.jpg"  # low cam

# === Checkerboard spec ===
CHECKERBOARD = (8, 6)     # inner corners (cols, rows)
SQUARE_SIZE_M = 0.028     # meters per square edge (e.g., 24 mm -> 0.024)

# === Intrinsics ===
# side (cam 0)
K_CAM0 = np.array([[509.05229773, 0.0, 333.59489432],
                   [0.0, 509.21723083, 240.21732484],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_CAM0 = np.array([ 0.03649518,  0.1122806 ,  0.00112988,  0.00355613, -0.64543531], dtype=np.float64)

# low (cam 2)
K_CAM2 = np.array([[523.99884119, 0.0, 325.88932755],
                   [0.0, 521.55379105, 238.28787204],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_CAM2 = np.array([ 0.15703463, -0.39054077, -0.00710589, -0.00253287,  0.24571638], dtype=np.float64)

def make_objpoints(board, square_size_m):
    cols, rows = board
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_size_m

def find_corners(img_path, board):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(
        gray, board,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        raise RuntimeError(f"Checkerboard not found in {img_path}")
    corners = cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    )
    return img, corners

def solve_one(img_path, K, dist, board, square_size_m, cam_name):
    img, corners = find_corners(img_path, board)
    objp = make_objpoints(board, square_size_m)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError(f"solvePnP failed for {cam_name}")

    R, _ = cv2.Rodrigues(rvec)

    # Reprojection error
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = np.linalg.norm((proj.reshape(-1,2) - corners.reshape(-1,2)), axis=1).mean()

    # Plane n^T X + d = 0 from checkerboard pose in camera coords
    n = R[:, 2]
    d = -float(n @ tvec.reshape(3))

    out = {
        "image": img_path,
        "camera_matrix": K.tolist(),
        "distortion": dist.tolist(),
        "rvec": rvec.ravel().tolist(),
        "tvec_m": tvec.ravel().tolist(),
        "R": R.tolist(),
        "plane_normal": n.tolist(),
        "plane_d": d,
        "reproj_error_px": float(err),
        "resolution": {"w": int(img.shape[1]), "h": int(img.shape[0])},
        "checkerboard": {"cols": board[0], "rows": board[1], "square_size_m": square_size_m},
    }
    out_path = Path(f"extrinsics_{cam_name}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n== {cam_name} ==  reprojection error: {err:.2f} px  -> saved {out_path}")
    return R, tvec

def main():
    solve_one(IMG_CAM0, K_CAM0, DIST_CAM0, CHECKERBOARD, SQUARE_SIZE_M, "cam0_side")
    solve_one(IMG_CAM2, K_CAM2, DIST_CAM2, CHECKERBOARD, SQUARE_SIZE_M, "cam2_low")

if __name__ == "__main__":
    main()

