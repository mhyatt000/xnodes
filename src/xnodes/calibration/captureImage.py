import cv2

cap = cv2.VideoCapture(0)  # change index if multiple cameras
i = 0
while True:
    ret, frame = cap.read()
    frame.reshape
    if not ret:
        break
    cv2.imshow("Press SPACE to capture", frame)
    key = cv2.waitKey(1)
    if key == 32:  # spacebar
        cv2.imwrite(f"images/img_{i:02d}.jpg", frame)
        print(f"Saved images/img_{i:02d}.jpg")
        i += 1
    elif key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
