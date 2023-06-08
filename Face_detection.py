import cv2
import numpy as np

faceCascade=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
minArea=200
width=640
height=480
count=0

def getFace(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(imgGray,1.1,4)

    for (x,y,w,h) in faces:
        area=w*h
        if area>minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,'Face',(x+(w//2)-50,y+(h//2)-50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,0),1)
        return x,y,w,h

cam=cv2.VideoCapture(1)

cam.set(3,width)
cam.set(4,height)
cam.set(10,100)

while True:
    success,img=cam.read()
    x,y,w,h=getFace(img)
    imgRoi=img[y:y+h,x:x+w]
    cv2.imshow('Face',imgRoi)
    cv2.imshow('Video',img)
    if cv2.waitKey(1) and 0xFF==ord('a'):
        cv2.imwrite('pythonProject/Facescut/Face_'+str(count)+'.jpg',imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                    2, (0, 0, 255), 2)
        cv2.imshow('Video', img)
        cv2.waitKey(500)
        count+=1