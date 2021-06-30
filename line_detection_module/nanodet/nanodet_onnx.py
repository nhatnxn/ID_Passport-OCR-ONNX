from line_detection_module.nanodet.utils import detect_box
from numpy.lib.utils import info
from card_detection_module.nanodet.utils import aligh
import numpy as np
import cv2
from .utils import NanoDetONNX, get_box

IMAGE_SIZE  = (416, 416)
ONNX_PATH   =  ''

def detect_line(img):
    """

    Args:
        im (np.array): input image 

    Returns:
        dict: 
            {
                'address_line_1':   [left,top,right,bottom]
                'address_line_2':   [left,top,right,bottom]
                'birthday':         [left,top,right,bottom]
                'hometown_line_1':  [left,top,right,bottom]
                'hometown_line_2':  [left,top,right,bottom]
                'id':               [left,top,right,bottom]
                'name':             [left,top,right,bottom]
                'nation':           [left,top,right,bottom]
                'sex':              [left,top,right,bottom]
            }

    """

    detector = NanoDetONNX(ONNX_PATH, input_shape=IMAGE_SIZE)
    bbox, _, _ = detector.detect(img)
    
    info = get_box(bbox)
    return info

if __name__ == '__main__':
    img = cv2.imread('')
    info = detect_line(img)
