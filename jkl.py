import cv2
from cvzone.HandTrackingModule import HandDetector
from  cvzone.ClassificationModule import Classifier
import numpy as np
import time
import math
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model4/keras_model.h5","Model4/labels.txt")
offset = 20
imgSize = 300
folder = "Data/A"
counter = 0

labels = ["J","K","L"]
while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)
    imgOutout = img.copy()
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            else:
                continue
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            else:
                continue
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutout, (x - offset, y - offset - 50),
                      (x + offset + 180, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutout, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutout, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
    cv2.imshow("Image", imgOutout)
    cv2.waitKey(1)
