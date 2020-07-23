import cv2
import easygui

file_path = easygui.fileopenbox()

images =file_path

img = cv2.imread(images)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Face_Cascade_for_eyes = cv2.CascadeClassifier("haarcascade_eye.xml")# This one for eye detecting

eyes = Face_Cascade_for_eyes.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

for x,y,w,h in eyes:
    img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,25,0),5)

Face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")# this one for face detecting

faces = Face_Cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),5)


cv2.imshow(images, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
