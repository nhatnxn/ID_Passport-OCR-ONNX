import time

from numpy.lib.utils import info
import torch
import numpy as np
import os
import onnxruntime
import torchvision
import cv2
from .utils import detect_batch_frame, polygon_from_corners, increase_border, get_dewarped_table


IMAGE_SIZE  = (640, 640)
ONNX_PATH   =  ''

def detect_line(img):
    """detect 9 information box

    Args:
        im (np.array): input image 

    Returns:
        dict:
            {
                'address_line_1':   [x,y,w,h]
                'address_line_2':   [x,y,w,h]
                'birthday':         [x,y,w,h]
                'hometown_line_1':  [x,y,w,h]
                'hometown_line_2':  [x,y,w,h]
                'id':               [x,y,w,h]
                'name':             [x,y,w,h]
                'nation':           [x,y,w,h]
                'sex':              [x,y,w,h]
            } 

    """
    PADDING_SIZE    = 8
    info         = {}
    list_dict       = ['address_line_1','address_line_2', 'birthday', 'hometown_line_1', 'hometown_line_2', 'id', 'name', 'nation', 'sex', 'passport']
 
    # check input
    if im is None:
        return []

    provider = os.getenv('PROVIDER', 'CPUExecutionProvider')
    model = onnxruntime.InferenceSession(ONNX_PATH, providers=[provider])
    target = detect_batch_frame(model, img, image_size=IMAGE_SIZE)
    
    if target is None:
        return []

    pts = polygon_from_corners(target)
    
    for i, inf in enumerate(pts):
        info[list_dict[i]] = inf

    return info


if __name__ == '__main__':
    im = cv2.imread('')
    info = detect_line(im)

        