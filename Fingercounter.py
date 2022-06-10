import HandtrackingModule as htm
import numpy as np
import cv2 as cv
import mediapipe as mp
import os


cap = cv.VideoCapture(0)
hd = htm.handDetector()

# keep track of the points of the tips of our fingers
tipids = [4, 8, 12, 16, 20]

while True:
    isTrue, frame = cap.read()
    #we'll draw our counting over this blank image 
    blank = np.zeros((100, 100, 1), dtype='uint8')
    frame, _ = hd.findHands(frame)
    lmlist1 = hd.ListLandmarks(frame)    #takes the landmarks of the points for the first hand
    lmlist2 = hd.ListLandmarks(frame, 1) #takes the landmarks of the second hand
    llmlist = [] #initialize a list that we'll loop over to get both hands through.
    if len(lmlist1) != 0 or len(lmlist2) != 0:  #check to see if the model has found any hands
        if lmlist1[1] == lmlist2[1]:  # this is to prevent the model of giving back the same hand twice
            llmlist.append(lmlist1)
        else:
            llmlist.append(lmlist1)
            llmlist.append(lmlist2)
    fingerups = []   # We'll use this to count our fingers
    for listl in llmlist:    #we initialize the loop over a hand at a time.
        if len(listl) != 0:
            #iterate over the tips of the fingers and check if their distance to the point in the center of the hand is smaller than a threshold
            #if its not we'll append True to the fingerups list.
            for id in tipids:  
                point1 = np.array([listl[0][1], listl[0][2]])
                point2 = [listl[id][1], listl[id][2]]
                distance = np.linalg.norm((point1-point2))
                if distance < 120:
                    print('finger closed')

                else:
                    print('finger open')
                    fingerups.append(True)  

    #takes the length of the fingerups and draw it over the blank image
    
    #cv.putText(blank, f'{len(fingerups)}', (37, 60),cv.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)
    
    #takes the length of the fingerups and draw it on our image
    cv.putText(frame, f'{len(fingerups)}', (frame.shape[0]//6, frame.shape[1]//6),
               cv.FONT_HERSHEY_COMPLEX, 2.5, (255, 0, 0), 3)
    #uses the blank image as a background to our counter.

    #frame[0:100, 0:100] = blank
    cv.imshow('frame', frame)
    if cv.waitKey(2) & 0xff == ord('d'):
        break
