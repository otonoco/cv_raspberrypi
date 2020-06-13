import cv2
import numpy as np
import os 
import datetime

import threading

import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email.MIMEBase import MIMEBase
from email import Encoders

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


# create authentication for google drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)


# open log file
logfile = open("log.txt", "a", 0)
upload_log = drive.CreateFile({'title': 'log.txt'})


# create facial recognizer and train on learned faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainedData/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

# Initializations
names = ['bruh', 'Bill', 'Albert', 'BTANG']
filename = 'videoData/surveillance'
fps = 10.0
res = '480p'
alertEnabled = 1
alertAll = 1 # 1: alert for all people 0: alert only for unknown people

id = 0
cam = cv2.VideoCapture(0)

# Changes video resolution
def change_res(cam, width, height):
    cam.set(3, width)
    cam.set(4, height)

# Standard Video Dimensions Sizes
DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cam, res='480p'):
    w, h = DIMENSIONS["480p"]
    if res in DIMENSIONS:
        w, h = DIMENSIONS[res]

    change_res(cam, w, h)
    return w, h


vidCount = 0
out = cv2.VideoWriter(filename + str(vidCount) + '.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, get_dims(cam, res))

# minimum window size that can be a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', port=465)
    server.ehlo()
    server.login('iotcfinalproject@gmail.com','@iotc1234')
except:
    print ("Something went wrong with server login")

def email_alert(filename):
    msg = MIMEMultipart()
    
    sender = "Rasperry Pi <iotcfinalproject@gmail.com>"
    recipients = ['albertlin97@gmail.com']
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = "IOTC Unknown Face Detected"
    msg.attach(MIMEText("""Unknown User Detected from security system"""))
    
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(filename, "rb").read())
    Encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(filename))
    msg.attach(part)
    
    try:
        server.sendmail(sender, recipients, msg.as_string())
        print("Alert Succesfully Sent")
    except:
        print("Failed to send Alert")
        return server

def update_and_upload(f, upload_log, logfile, drive):
     #upload to google drive
    upload_video = drive.CreateFile({'title': f })
    upload_video.SetContentFile(filename + str(vidCount) + '.avi')
    upload_video.Upload()
        

    # write to log of people
    logfile.write(time_right_now + " " + id_name + " " + filename + str(vidCount) + '.avi' + "\n")
    upload_log.SetContentFile("log.txt")
    upload_log.Upload()



test = 0
videoRecording = 0
recFrames = 0
frameSkipBuffer = 3
frameBuffer = 0
t1 = None
t2 = None

while True:
        
    # get camera image
    ret, img =cam.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2,
        minNeighbors = 5, minSize = (int(minW), int(minH)),)

    t = str(datetime.datetime.now())
    time_right_now = t[:t.find(".")]
    
    # write text to camera
    cv2.putText(img, time_right_now, (10,20), 1, 1, (255,255,255), 1)
    knownFaceDetected = 0
    faceDetected = 0

    # detect faces and predict who it is
    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # confidence -> better matches are closer to 0
        if (confidence < 75):
            id_name = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            knownFaceDetected = 1
            
        else:
            id_name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            faceDetected = 1
        
        # print labels to screen along with accuracy
        cv2.putText(img, id_name, (x+5,y-5), 3, 1, (255,255,255), 2)
        cv2.putText(img, "Acc:" + str(confidence), (x+5,y+h-5), 3, 1, (255,255,0), 1)  
    
    if t1 is not None:
        t1.join()
        t1 = None
    if t2 is not None:
        t2.join()
        t2 = None


    # start recording if a face is seen
    if (faceDetected == 0 and videoRecording == 1):
        frameBuffer += 1
        if (frameBuffer >= frameSkipBuffer):
            #end recording and save
            if (recFrames > 10):
                vidCount += 1
                out = cv2.VideoWriter(filename + str(vidCount) + '.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, get_dims(cam, res))

                # only send alerts for unknown people unless alertAll is on
                if ( alertEnabled and ((alertAll == 0 and knownFaceDetected == 1) or alertAll == 1)):
                    # email_alert(filename + str(vidCount - 1) + '.avi')
                    t1 = threading.Thread(target=email_alert, args=(filename + str(vidCount - 1) + '.avi',))
                    t1.start()
                    print("email alert sent!")
            else:
                out = cv2.VideoWriter(filename + str(vidCount) + '.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, get_dims(cam, res))


            t2 = threading.Thread(target=update_and_upload, args=(filename + str(vidCount) + '.avi', upload_log, logfile, drive,))
            t2.start()
            # update_and_upload(filename + str(vidCount) + '.avi', upload_log, logfile, drive)




            videoRecording = 0
            recFrames = 0
        
    if (faceDetected == 1):
        frameBuffer = 0
        videoRecording = 1
        recFrames+=1
        out.write(img)
        
    
    cv2.imshow('camera',img) 
    #press esc to close
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break


server.quit()
print("Quit Server")
print("Exit")
out.release()
cam.release()
cv2.destroyAllWindows()
