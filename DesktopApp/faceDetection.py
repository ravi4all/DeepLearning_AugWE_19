import cv2

def showCamera():
    dataset = cv2.CascadeClassifier('data.xml')

    cap = cv2.VideoCapture(0)

    while True:
        boolean, image = cap.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        for x,y,w,h in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,255),5)
        # cv2.imshow('result',image)
        cv2.imshow('result', gray)
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()