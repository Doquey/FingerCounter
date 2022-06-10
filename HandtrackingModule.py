import cv2 as cv
from cv2 import waitKey
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import time

# we'll create a handDetector class that can be then instatiated and used to detect hands and return landmarks values


class handDetector():

    # we initialize the detector: notice that we need these parameters because we'll use then we instantiating the hands class from mediapipe
    # we initialize the class with the parameters of the hands of the mediapipe
    def __init__(self, mode=False, max_hands=2, min_det_con=0.5, min_track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.min_det_con = min_det_con
        self.min_track_con = min_track_con
        self.mpHands = mp.solutions.hands
        # instantiate the hands mediapipe module with the users parameters
        self.hands = self.mpHands.Hands(
            self.mode, self.max_hands, min_detection_confidence=self.min_det_con, min_tracking_confidence=self.min_track_con)
        # instantiate a drawing util to our model
        self.mpDraw = mp.solutions.drawing_utils

    # having initilized the detector we can use its properties and itself to make detections with our model:
    # this function will be called as a method from this class and, it'll use the class element that is the detector
    # to make detections in a specific img

    def findHands(self, frame, draw=True):
        framex = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(framex)
        if self.results.multi_hand_landmarks is not None:
            for self.handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, self.handlms, self.mpHands.HAND_CONNECTIONS)
        return frame, self.results.multi_hand_landmarks

    def ListLandmarks(self, frame, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) != 1:
                # get the landmarks for the specific hand we want.
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape  # we get the shape of our image.
                    # Remember that the landmarks of each of those points in our hands is in decimal value, we can then multiply then to get the portion of the height or width of our image the point is at in pixels.
                    cx, cy = int(lm.x * w), int(lm.y*h)
                    lmlist.append([id, cx, cy])
                    # we can do this to check which landmark corresponds to each point in our hand
                    if draw:
                        cv.putText(frame, f'{id}', (cx, cy),
                                   cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                        cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            else:
                myHand = self.results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape  # we get the shape of our image.
                    # Remember that the landmarks of each of those points in our hands is in decimal value, we can then multiply then to get the portion of the height or width of our image the point is at in pixels.
                    cx, cy = int(lm.x * w), int(lm.y*h)
                    lmlist.append([id, cx, cy])
                    # we can do this to check which landmark corresponds to each point in our hand
                    if draw:
                        cv.putText(frame, f'{id}', (cx, cy),
                                   cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                        cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        return lmlist


# boilerplate code to show off what our module can do.
def main():

    capture = cv.VideoCapture(0)
    cTime = 0
    preTime = 0
    detector = handDetector()
    while True:
        isTrue, frame = capture.read()
        frame, _ = detector.findHands(frame)
        listlm = detector.ListLandmarks(frame, 0)

        cTime = time.time()
        # we calculate the fps(frames per second). this is usefull to see how fast is our process going.
        fps = 1/(cTime-preTime)
        preTime = cTime

        cv.putText(frame, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        cv.imshow('img', frame)
        cv.waitKey(1)


# this says that if we are running the module in the main file, it'll run the code in main, otherwise it'll run the code outside of it.
if __name__ == '__main__':
    main()
