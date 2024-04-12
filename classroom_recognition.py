import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haarcascade_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

train_DIR = r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Train'

list_person = os.listdir(train_DIR)
print(f'\nLista wszystkich osob: {list_person}\n')

def test_class():

    # Tu definiujemy imiona osob przypisanych do danych grup.
    grupa_1 = ['Klaudia', 'Jakub', 'Konrad', 'Arkadiusz', 'Maciek']
    grupa_2 = ['Szymon', 'Marta', 'Sebastian', 'Patryk', 'Oliwia']

    # W tym katalogu mamy zdjecia osob, ktore przyszly na zajecia.
    test_DIR = r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Classroom1'
    list_person_detected = []

    for root, dirs, files in os.walk(test_DIR):
        for file in files:

            img_path = os.path.join(root, file)
            img_read = cv.imread(img_path)

            img = cv.resize(img_read, (900,1200), interpolation=cv.INTER_AREA)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            face_detect = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in face_detect:
                face = img_gray[y:y + h, x:x + w]

                face = cv.resize(face, (500, 500), interpolation=cv.INTER_AREA)

                label, confidence = face_recognizer.predict(face)

                if confidence < 35:
                    print(f'Etykieta = {list_person[label]}, z pewnoscia: {confidence}')
                    list_person_detected.append(list_person[label])

                    cv.putText(img, str(list_person[label]), (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 127, 255), thickness=2)
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 127), thickness=3)
                else:
                    print('!-- Przedstawiony wynik moze byc niepoprawny, wymagana dodatkowa weryfikacja --!')
                    print(f'Etykieta = {list_person[label]}, z pewnoscia: {confidence}\n')
                    cv.putText(img, str(list_person[label]), (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 127, 255), thickness=2)
                    cv.putText(img, ('!-- ZWERYFIKUJ WYNIK --!'), (x, y - 60), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 127), thickness=3)

            img_print = cv.resize(img, (585, 780), interpolation=cv.INTER_AREA)
            cv.imshow('Rozpoznana twarz', img_print)
            cv.waitKey(0)

    print(f'\nLista wykrytych osob: {list_person_detected}')

    print('\n ---Lista obecnosci grupy 1---')
    for x in range(len(grupa_1)):
        if grupa_1[x] in list_person_detected:
            print(f'{x+1}) {grupa_1[x]}: obecny/a')
        else:
            print(f'{x+1}) {grupa_1[x]}: nieobecny/a')

    print('\n ---Lista obecnosci grupy 2---')
    for x in range(len(grupa_2)):
        if grupa_2[x] in list_person_detected:
            print(f'{x+1}) {grupa_2[x]}: obecny/a')
        else:
            print(f'{x+1}) {grupa_2[x]}: nieobecny/a')

test_class()



