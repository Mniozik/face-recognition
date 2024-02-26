import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade_face.xml')

def test_function():

    img_read = cv.imread(r'C:\Users\Konrad\PycharmProjects\Projekt1\Inzynierka_ALL\Photos\Train\Arkadiusz\323079806_2576644772487872_7661886006962807307_n.jpg')

    print(f'{img_read.shape[1],img_read.shape[0]}')
    img = cv.resize(img_read, (900, 1200), interpolation=cv.INTER_AREA)

    print(f'{img.shape[1],img.shape[0]}')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_detect = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y , w, h) in face_detect:
        face = img_gray[y:y+h, x:x+w]
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 127), thickness=3)

    img_print = cv.resize(img, (450, 600), interpolation=cv.INTER_AREA)
    cv.imshow('Zlokalizowana twarz', img_print)
    cv.imshow('Wyodrebniona twarz', face)
    cv.waitKey(0)

test_function()


