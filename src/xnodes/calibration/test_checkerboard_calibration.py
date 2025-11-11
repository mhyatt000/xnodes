# mini_triangulate_checkerboard.py
import json, cv2, numpy as np

# edit these
CAM1_JSON = "extrinsics_cam0_side.json"
CAM2_JSON = "extrinsics_cam2_low.json"
IMG1 = "checkerboard_captures/cam0_00_1762205304731.jpg"
IMG2 = "checkerboard_captures/cam2_00_1762205304731.jpg"
CHECKERBOARD = (8, 6)  # inner corners (cols, rows)

def load_cam(p):
    d = json.load(open(p))
    K = np.array(d["camera_matrix"], np.float64)
    dist = np.array(d["distortion"], np.float64).reshape(-1)
    R = np.array(d["R"], np.float64)
    t = np.array(d["tvec_m"], np.float64).reshape(3,1)
    P = K @ np.hstack([R, t])
    return K, dist, P

def find_corners(img_path):
    img = cv2.imread(img_path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCorners(gray, CHECKERBOARD)
    if not ok: raise RuntimeError(f"board not found in {img_path}")
    c = cv2.cornerSubPix(gray, c, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    return c.reshape(-1,2)

K1, d1, P1 = load_cam(CAM1_JSON)
K2, d2, P2 = load_cam(CAM2_JSON)

p1 = find_corners(IMG1)
p2 = find_corners(IMG2)

# undistort to match K
p1u = cv2.undistortPoints(p1.reshape(-1,1,2), K1, d1, P=K1).reshape(-1,2)
p2u = cv2.undistortPoints(p2.reshape(-1,1,2), K2, d2, P=K2).reshape(-1,2)

X4 = cv2.triangulatePoints(P1, P2, p1u.T, p2u.T)     # 4xN
X3 = (X4[:3] / X4[3]).T                               # Nx3 in world frame (meters)

z = X3[:,2]
print("First 5 points:\n", X3[:5])
print(f"Depth stats (m): min={z.min():.4f} max={z.max():.4f} std={z.std():.6f}")

