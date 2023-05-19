import cv2 as cv
import numpy as np
import face_recognition
import os

path = 'Images'
images = []
class_names = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)
    class_names.append(os.path.splitext(cls)[0])
print(class_names)

def findEncodings(images):
    encodeList = []
    for img in images:
       img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
       encode = face_recognition.face_encodings(img)[0]
       encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")

cap = cv.VideoCapture(0)

while True:
    success,img = cap.read()
    img = cv.flip(img, 1)
    imgS = cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS,cv.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        print(matches)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matchIndex)

        if matches[matchIndex]:
            name = class_names[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv.imshow('webcam', img)
    if cv.waitKey(35) & 0xff == ord('f'):
        break
cap.release()
cv.destroyAllWindows()



