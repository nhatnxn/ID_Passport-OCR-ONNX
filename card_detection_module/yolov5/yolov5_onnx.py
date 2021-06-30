import time
import torch
import numpy as np
import os
import onnxruntime
import torchvision
import cv2
from .utils import detect_batch_frame, polygon_from_corners, increase_border, get_dewarped_table


IMAGE_SIZE  = (640, 640)
ONNX_PATH   =  ''

def detect_table_corners(img):
    
    PADDING_SIZE    = 8
    corners         = []

    # check input
    if im is None:
        return []

    provider = os.getenv('PROVIDER', 'CPUExecutionProvider')
    model = onnxruntime.InferenceSession(ONNX_PATH, providers=[provider])
    target = detect_batch_frame(model, img, image_size=IMAGE_SIZE)
    
    if target is None:
        return []

    pts = polygon_from_corners(target)
    if pts is None:
        return []

    pts = pts.astype(int)
    
    if pts is None:
        return []
    else:
        corners = increase_border(pts, PADDING_SIZE)
        corners = [(int(p[0]), int(p[1])) for p in corners]
    
    return corners

def detect_card(im):
    # call api 1: detect corners
    # Returns a list: 
    #       - case 1: table detected: [top-left, top-right, bottom-right, bottom-left]
    #                 e.g. [(1,2), (3, 4), (5, 6), (7, 8)]
    #       - case 2: no table detected: []
    corners = detect_table_corners(im)
    # call api 2: dewarp table with corners:
        # return:
        # - case 1: dewarped image
        # - case 2: None
    if len(corners) == 4:
        image_aligh = get_dewarped_table(im,corners)
        return image_aligh, True
    else:
        return im, False

if __name__ == '__main__':
    im = cv2.imread('')
    img, status = detect_card(im)
    print(img, status)

        