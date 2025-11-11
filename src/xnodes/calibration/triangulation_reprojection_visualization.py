import json, cv2, numpy as np

def load_cam(p):
    d = json.load(open(p)); K=np.array(d["camera_matrix"]); dist=np.array(d["distortion"]).ravel()
    R=np.array(d["R"]); t=np.array(d["tvec_m"]).reshape(3,1)
    rvec,_=cv2.Rodrigues(R); return K,dist,rvec,t

# X_world = Nx3 from your triangulation
K1,dist1,rvec1,t1 = load_cam("extrinsics_cam0_side.json")
K2,dist2,rvec2,t2 = load_cam("extrinsics_cam2_low.json")

def proj_err(img_path, K, dist, rvec, tvec, Xw, corners_px):
    img = cv2.imread(img_path)
    pts,_ = cv2.projectPoints(Xw.astype(np.float64), rvec, tvec, K, dist)
    pts = pts.reshape(-1,2)
    err = np.linalg.norm(pts - corners_px, axis=1).mean()
    for p in pts.astype(int): cv2.circle(img, tuple(p), 2, (0,255,0), -1)
    for p in corners_px.astype(int): cv2.circle(img, tuple(p), 2, (0,0,255), -1)
    cv2.imshow("reproj check", img); cv2.waitKey(0); cv2.destroyAllWindows()
    return err

# corners1, corners2 are the 2D corner arrays you detected earlier (Nx2)
e1 = proj_err("checkerboard_captures/cam0_00_1762205304731.jpg", K1, dist1, rvec1, t1, X_world, corners1)
e2 = proj_err("checkerboard_captures/cam2_00_1762205304731.jpg", K2, dist2, rvec2, t2, X_world, corners2)
print(f"Reproj error cam0: {e1:.2f}px, cam2: {e2:.2f}px")

