from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import cv2

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    results = rcnn.detect([img], verbose=0)
    obj = results[0]['rois']
    for x,y,w,h in obj:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),4)

    cv2.imshow('result',img)
    if cv2.waitKey(4) == 27:
        break

cap.release()
cv2.destroyAllWindows()
