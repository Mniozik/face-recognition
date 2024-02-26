import cv2 as cv
import os
import numpy as np

haar_cascade = cv.CascadeClassifier('haarcascade_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

train_DIR = r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Train'

list_person = os.listdir(train_DIR)
print(f'\nLista wszystkich osob: {list_person}\n')

def test_function():

    img_read = cv.imread(r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Test\Konrad\322134026_700969828209058_1608650002465852454_n.jpg')

    print(f'{img_read.shape[1],img_read.shape[0]}')
    img = cv.resize(img_read, (900, 1200), interpolation=cv.INTER_AREA)
    print(f'{img.shape[1],img.shape[0]}')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_detect = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y , w, h) in face_detect:
        face = img_gray[y:y+h, x:x+w]

        face = cv.resize(face, (500, 500), interpolation=cv.INTER_AREA) #wprowadzenie, by kazda twarz byla taka sama

        label, confidence = face_recognizer.predict(face)
        print(f'Etykieta = {list_person[label]}, z pewnoscia: {confidence}')

        cv.putText(img, str(list_person[label]), (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 127, 255), thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 127), thickness=3)

    cv.imshow('Wykryta twarz', img)
    cv.waitKey(0)

test_function()


