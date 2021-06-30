from card_detection_module.nanodet.utils import aligh
import numpy as np
import cv2
from .utils import NanoDetONNX, AlighModel

IMAGE_SIZE  = (416, 416)
ONNX_PATH   =  ''

def detect_card(img):
    """detect 4 corners of a table

    Args:
        im (np.array): input image 

    Returns:
        list: 
            - case 1: table detected: [top-left, top-right, bottom-right, bottom-left]
            - case 2: no table detected: []
    """
    detector = NanoDetONNX(ONNX_PATH, input_shape=IMAGE_SIZE)
    bbox, _, _ = detector.detect(img)
    
    """dewarped table with conner

    Args:
        im (np.array): input image 

    Returns:
        case1: None, False
        case2: dewarped image, True
    """
    aligh_model = AlighModel()
    img_aligh, have_card = aligh_model.aligh(img, bbox)

    if not have_card:
        return img_aligh, False
    return img_aligh, True

if __name__ == '__main__':
    img = cv2.imread('')
    image_aligh, have_card = detect_card(img)
