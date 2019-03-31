import numpy as np
import cv2
from PIL import Image
import time
class GNet:
    def __init__(self):
        self.lable_file = "geogleNet_parameters/synset_words.txt"
        self.prototxt = "geogleNet_parameters/bvlc_googlenet.prototxt"
        self.model = "geogleNet_parameters/bvlc_googlenet.caffemodel"
        self.video_path="sample.mp4"
        self.cascPath = "faceDetect/haarcascade_frontalface_default.xml"

        print("[INFO] loading label...")
        rows = open(self.lable_file).read().strip().split("\n")
        self.classes = [row[row.find(" ") + 1:].split(",")[0] for row in rows]

        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

        print("[INFO] loading video...")
        self.video_capture = cv2.VideoCapture(self.video_path)

        print("[INFO] loading faceCascade...")
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)

    def trainAndDetect(self):
        if self.video_capture.isOpened():
            success = True
        else:
            success = False
            print("Reading file fails!")
        while success:
            success, frame = self.video_capture.read()
            # training by geogleNet starts
            blob = cv2.dnn.blobFromImage(frame, 1, (224, 224))
            self.net.setInput(blob)
            preds=self.net.forward()
            # get the top five of predicting result
            rstIndex=np.argsort(preds[0])[::-1][:3]
            for (i, idx) in enumerate(rstIndex):
                if i == 0:
                    text = "Label: {}, {:.2f}%".format(self.classes[idx],preds[0][idx] * 100)
                    cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
                print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,self.classes[idx], preds[0][idx]))
            # training by geogleNet ends
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # face detect starts
            cv2.imshow("Image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print

    def __del__(self):
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    geogleNet=GNet()
    geogleNet.trainAndDetect()