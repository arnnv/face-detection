import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=min_detection_confidence,
                                                                model_selection=model_selection)
        self.results = None

    def findFaces(self, img, draw=True):
        img = cv2.resize(img, (1200, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []

        if self.results.detections:
            for i, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                bboxs.append([bbox, detection.score])

                if draw:
                    cv2.rectangle(img, bbox, (0, 255, 0), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        return img, bboxs


def main():
    cap = cv2.VideoCapture('videos/1.mp4')
    pTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
