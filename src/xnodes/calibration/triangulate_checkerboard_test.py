# mini_triangulate_checkerboard.py
import json, cv2, numpy as np

CAM1_JSON = "extrinsics_cam0_side.json"
CAM2_JSON = "extrinsics_cam2_low.json"
IMG1 = "checkerboard_captures/cam0_00_1762205304731.jpg"
IMG2 = "checkerboard_captures/cam2_00_1762205304731.jpg"
CHECKERBOARD = (8, 6)

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

p1u = cv2.undistortPoints(p1.reshape(-1,1,2), K1, d1, P=K1).reshape(-1,2)
p2u = cv2.undistortPoints(p2.reshape(-1,1,2), K2, d2, P=K2).reshape(-1,2)

X4 = cv2.triangulatePoints(P1, P2, p1u.T, p2u.T)
X3 = (X4[:3] / X4[3]).T

z = X3[:,2]
print("First 5 points:\n", X3[:5])
print(f"Depth stats (m): min={z.min():.4f} max={z.max():.4f} std={z.std():.6f}")
np.save("X_world.npy", X3)

# quick reprojection error check
def load_full(p):
    d=json.load(open(p)); K=np.array(d["camera_matrix"]); dist=np.array(d["distortion"]).ravel()
    R=np.array(d["R"]); t=np.array(d["tvec_m"]).reshape(3,1); rvec,_=cv2.Rodrigues(R)
    return K,dist,rvec,t

K1f,d1f,rvec1,t1 = load_full(CAM1_JSON)
K2f,d2f,rvec2,t2 = load_full(CAM2_JSON)
proj1,_ = cv2.projectPoints(X3, rvec1, t1, K1f, d1f)
proj2,_ = cv2.projectPoints(X3, rvec2, t2, K2f, d2f)
e1 = np.linalg.norm(proj1.reshape(-1,2) - p1, axis=1).mean()
e2 = np.linalg.norm(proj2.reshape(-1,2) - p2, axis=1).mean()
print(f"Reproj error cam0_side: {e1:.2f}px  cam2_low: {e2:.2f}px")

# --- Visual overlay & save ---
def overlay(img_path, K, dist, rvec, tvec, X3, detected_px, out_path):
    img = cv2.imread(img_path).copy()
    proj, _ = cv2.projectPoints(X3.astype(np.float64), rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2).astype(int)
    det  = detected_px.astype(int)

    # green = reprojected, red = detected
    for p in proj: cv2.circle(img, tuple(p), 2, (0,255,0), -1)
    for p in det:  cv2.circle(img, tuple(p), 2, (0,0,255), -1)

    cv2.imwrite(out_path, img)
    print(f"saved {out_path}")

# reuse K1f,d1f,rvec1,t1, K2f,d2f,rvec2,t2, plus p1,p2 and X3 from your script
overlay(IMG1, K1f, d1f, rvec1, t1, X3, p1, "cam0_overlay.png")
overlay(IMG2, K2f, d2f, rvec2, t2, X3, p2, "cam2_overlay.png")

