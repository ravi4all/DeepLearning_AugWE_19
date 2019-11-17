from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def draw_image_with_boxes(filename, boxes_list):
     data = pyplot.imread(filename)
     pyplot.imshow(data)
     ax = pyplot.gca()
     for box in boxes_list:
          y1, x1, y2, x2 = box
          width, height = x2 - x1, y2 - y1
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          ax.add_patch(rect)
     pyplot.show()

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
img = load_img('img_2.jpg')
img = img_to_array(img)
results = rcnn.detect([img], verbose=0)
draw_image_with_boxes('img_2.jpg', results[0]['rois'])
