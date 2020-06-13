import cv2
import os

face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_id = input('\n User ID: ')
print("\n Starting Face Recognition")

count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, -1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w,y + h), (255, 0, 0), 2)     
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y : y + h, x : x + w])
        cv2.imshow('image', img)
    
    #press esc or wait for 30 pictures to exit
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    elif count >= 30:
         break

print("\n Finished Face Recognition")
cam.release()
cv2.destroyAllWindows()

