import cv2
import time
from pathlib import Path

SAVE_DIR = Path("checkerboard_captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Open both cameras
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# (Optional) lock resolution if you want
for cap in (cap0, cap2):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

i = 0
while True:
    ok0, f0 = cap0.read()
    ok2, f2 = cap2.read()
    if not ok0 or not ok2:
        print("Failed to read from one of the cameras.")
        break

    cv2.imshow("Cam0 (low?)", f0)
    cv2.imshow("Cam2 (side?)", f2)

    k = cv2.waitKey(1) & 0xFF
    if k == 32:  # SPACE to capture
        ts = int(time.time() * 1000)
        p0 = SAVE_DIR / f"cam0_{i:02d}_{ts}.jpg"
        p2 = SAVE_DIR / f"cam2_{i:02d}_{ts}.jpg"
        cv2.imwrite(str(p0), f0)
        cv2.imwrite(str(p2), f2)
        print(f"Saved:\n  {p0}\n  {p2}")
        i += 1
    elif k == 27:  # ESC to quit
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()

