import HandtrackingModule as htm
import numpy as np
import cv2 as cv
import mediapipe as mp
import os


cap = cv.VideoCapture(0)
hd = htm.handDetector()

tipids = [4, 8, 12, 16, 20]

while True:
    isTrue, frame = cap.read()
    blank = np.zeros((100, 100, 1), dtype='uint8')
    frame, _ = hd.findHands(frame)
    lmlist1 = hd.ListLandmarks(frame)
    lmlist2 = hd.ListLandmarks(frame, 1)
    llmlist = []
    if len(lmlist1) != 0 or len(lmlist2) != 0:
        if lmlist1[1] == lmlist2[1]:
            llmlist.append(lmlist1)
        else:
            llmlist.append(lmlist1)
            llmlist.append(lmlist2)
    fingerups = []
    for listl in llmlist:
        if len(listl) != 0:
            for id in tipids:
                point1 = np.array([listl[0][1], listl[0][2]])
                point2 = [listl[id][1], listl[id][2]]
                distance = np.linalg.norm((point1-point2))
                if distance < 120:
                    print('finger closed')

                else:
                    print('finger open')
                    fingerups.append(True)

    # another method to know if a finger is closed or open would be :
    # if len(lmlist) != 0:
    #     for id in tipids:

    # if id != 4:
    #         if lmlist[id][2] < lmlist[(id-2)][2]:
    #             print('finger open')
    #     else:
    #         if lmlist[id][2] < lmlist[id-1][2]:
    #             print('finger open')
    # print('all fingers closed')
    cv.putText(blank, f'{len(fingerups)}', (37, 60),
               cv.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)
    frame[0:100, 0:100] = blank
    cv.imshow('frame', frame)
    if cv.waitKey(2) & 0xff == ord('d'):
        break
