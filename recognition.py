import face_recognition
import cv2 as cv
import numpy as np


imgMy = face_recognition.load_image_file("kumar.jpg")
imgMy = cv.cvtColor(imgMy,cv.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("kumar test.jpg")
imgTest = cv.cvtColor(imgTest,cv.COLOR_BGR2RGB)

def rescaleFrame(frame, scale = 0.25):
    #This Method will work for video , images and live Video
    width = int(frame.shape[1]* scale)
    height = int(frame.shape[0]* scale)
    dimensions = (width , height)

    return cv.resize(frame , dimensions, interpolation=cv.INTER_AREA )

imgMy = rescaleFrame(imgMy)
imgTest = rescaleFrame(imgTest)

#Finding faces and encodings
faceLoc = face_recognition.face_locations(imgMy)[0]
encodeMy = face_recognition.face_encodings(imgMy)[0]
#print(faceLoc)
#returns top right and bottom left
cv.rectangle(imgMy,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeMy],encodeTest)
faceDis = face_recognition.face_distance([encodeMy],encodeTest)
print(faceDis)
print(results)
cv.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv.imshow("Me",imgMy)
cv.imshow("Test One",imgTest)
cv.waitKey(0)