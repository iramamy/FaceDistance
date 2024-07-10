import cv2
import cvzone
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

W = 6.3
f = 700 # focal distance camera
alpha = 10 # sensitivity

text = 'This is a simple test \nabout dynamic text size \nbased on the distance of \nuser\'s face in  \nfront of a screen.'

texts = text.split('\n')

while True:
    success, image = cap.read()
    imgText = np.zeros_like(image)

    img, faces = detector.findFaceMesh(image, draw=False)

    if faces:
        face = faces[0]
        
        # Left, right eyes
        pointLeft = face[145]
        pointRight = face[374]
        
        w, _ = detector.findDistance(pointLeft, pointRight)

        ### Find distance
        distance = (W*f)/w

        cvzone.putTextRect(
            img,
            f'Depth {int(distance)}cm',
            (face[10][0]-100, face[10][1]-75),
            scale=2)

        
        for i, text in enumerate(texts):
            y0, dy = 50, 25 + int((int(distance/alpha)*alpha)/4)

            height = y0 + (i * dy)
            scale = 0.4 + (int(distance/alpha)*alpha)/75

            cv2.putText(imgText, text, (50, height), cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

    image_stacked = cvzone.stackImages([img, imgText], 2, 1)

    cv2.imshow("Image", image_stacked)
    cv2.waitKey(1)  

cap.release()
