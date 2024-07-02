import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('./videos/4540151-hd_1920_1080_30fps.mp4')

pTime = 0

while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
