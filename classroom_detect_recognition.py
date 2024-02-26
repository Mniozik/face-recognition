import cv2 as cv
import os
import numpy as np

haar_cascade = cv.CascadeClassifier('haarcascade_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

train_DIR = r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Train'

list_person = os.listdir(train_DIR)
print(f'List of trained person: {list_person}')

def test_class():

    #Zdefiniowanie kto powinien byc na jakich zajeciach
    grupa_T1 = ['Klaudia', 'Jakub', 'Konrad', 'Arkadiusz', 'Maciej']
    grupa_T2 = ['Szymon', 'Marta', 'Sebastian', 'Patryk', 'Oliwia']

    test_DIR = r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Classroom1'
    list_person_detected = []

    for root, dirs, files in os.walk(test_DIR): #tworzy sie lista files
        for file in files: #jesli znajduje jakies pliki (np .jpg [nie foldery]) to tworzy z nich liste files, i wyciagamy pojedynczo z niej

            img_path = os.path.join(root, file)
            img_read = cv.imread(img_path)

            img = cv.resize(img_read, (900,1200), interpolation=cv.INTER_AREA)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            face_detect = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

            # # Jesli wykryje wiecej twarzy niz 1
            # # Jesli nie chcemy by sprawdzalo pare twarzy na zdjeciu jednoczesnie (czesto ta druga twarz to zle wykryty element)
            # len_faces = len(face_detect)
            # if len_faces > 1:
            #     print(f'---Error---\nInput image is not correctly, detected {len_faces} faces\nPlease try again\n-----------')
            #     return 0

            for (x, y, w, h) in face_detect:
                face = img_gray[y:y + h, x:x + w]

                face = cv.resize(face, (500, 500), interpolation=cv.INTER_AREA) #wprowadzenie, by kazda twarz byla taka sama

                label, confidence = face_recognizer.predict(face)

                #Zeby odrzucalo zbyt ryzykowne wyniki
                if confidence < 40:
                    print(f'Label = {list_person[label]} with a confidence of {confidence}')
                    list_person_detected.append(list_person[label])

                    cv.putText(img, str(list_person[label]), (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 127, 255), thickness=2)
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 127), thickness=3)

            cv.imshow('Detected Face', img)
            cv.waitKey(0)

        print(f'List of recognized person: {list_person_detected}')


    # Wyniki programu

    print('\n ---Lista obecnosci grupy T1---')
    for x in range(len(grupa_T1)):
        if grupa_T1[x] in list_person_detected:
            print(f'{x+1}) {grupa_T1[x]}: obecny/a')
        else:
            print(f'{x+1}) {grupa_T1[x]}: nieobecny/a')

    print('\n ---Lista obecnosci grupy T2---')
    for x in range(len(grupa_T2)):
        if grupa_T2[x] in list_person_detected:
            print(f'{x+1}) {grupa_T2[x]}: obecny/a')
        else:
            print(f'{x+1}) {grupa_T2[x]}: nieobecny/a')

test_class()



