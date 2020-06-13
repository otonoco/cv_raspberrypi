import cv2
import numpy as np
import os 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainedData/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


id = 0
names = ['None', 'Bill', 'Albert', 'BTANG']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# minimum window size that can be a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img =cam.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2,
        minNeighbors = 5, minSize = (int(minW), int(minH)),)
        
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y : y + h,x : x + w])
        # confidence -> better matches are closer to 0
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x + 5, y - 5), 3, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5,y + h - 5), 3, 1, (255, 255, 0), 1)  
    
    #press esc to close
    cv2.imshow('camera',img)
    cv2.imshow('gray', gray) 
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n Exit")
cam.release()
cv2.destroyAllWindows()
