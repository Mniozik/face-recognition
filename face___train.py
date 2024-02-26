import cv2 as cv
import os
import numpy as np

haar_cascade = cv.CascadeClassifier('haarcascade_face.xml')

labels = []
features = []

train_DIR = r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Train'

list_person = os.listdir(train_DIR)
print(f'List_person: {list_person}')

def train_function():

    for root, dirs, files in os.walk(train_DIR): #tworzy sie lista files
        for file in files: #jesli znajduje jakies pliki (np .jpg [nie foldery]) to tworzy z nich liste files, i wyciagamy pojedynczo z niej

            temp_person = os.path.basename(root)
            label = list_person.index(temp_person)

            img_path = os.path.join(root, file)
            img_read = cv.imread(img_path)

            img = cv.resize(img_read, (900,1200), interpolation=cv.INTER_AREA)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            face_detect = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in face_detect:
                face = img_gray[y:y+h, x:x+w]

                face = cv.resize(face, (500, 500), interpolation=cv.INTER_AREA) #wprowadzenie, by kazda twarz byla taka sama

                features.append(face)
                labels.append(label)

train_function()

print(f'Lenght of the features: {len(features)}')
print(f'Lenght of the labels: {len(labels)}')

#Conversion to numpy array
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.write('face_trained.yml')




