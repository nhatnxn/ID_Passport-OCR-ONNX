import os
import sys
import cv2
import math
import numpy as np

def best_point(t):
    return t[t[:,-2].argsort()[-1]]

def polygon_from_corners(t):
    t0 = t[t[:,-1] == 0] # top-left points
    t1 = t[t[:,-1] == 1] # top-right points
    t2 = t[t[:,-1] == 2] # bottom-right points
    t3 = t[t[:,-1] == 3] # bottom-left points
    
    if t0.shape[0] == 0 or t1.shape[0] == 0 or t2.shape[0] == 0 or t3.shape[0] == 0:
        return None
    
    A = best_point(t0)
    B = best_point(t1)
    C = best_point(t2)
    D = best_point(t3)
    
    return np.stack((A, B, C, D))[:,2:4]

def calculate_abc(A, B, C, d):
    n1, n2 = B-A
    M = (A+B)/2
    IM = np.array([(-n2*d)/math.sqrt(n1**2+n2**2), (n1*d)/math.sqrt(n1**2+n2**2)])

    CM = M - C
    # print((np.matmul(CM, np.transpose(IM))))
    if (np.matmul(CM, np.transpose(IM))) > 0:
        IM = -IM
    I = M - IM
    a1, b1 = IM
    c1 = -np.matmul(I, np.transpose(IM))
    # print(n1, n2, I, M, IM)
    return a1, b1, c1

def get_intersection(a1, b1, c1, a2, b2, c2):
    if a1 == 0:
        y = -c1/b1
        if a2 == 0:
            return nan, nan
        else:
            x = (-c2 - b2*y)/a2
            return x, y
    else:
        c = a2/a1
        y = (c2-c*c1)/(b1*c-b2)
        x = (-c1-b1*y)/a1
        return x, y

def increase_border(points, d = 10):
    A, B, C, D = points
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    a1, b1, c1 = calculate_abc(A, B, C, d)
    a2, b2, c2 = calculate_abc(B, C, D, d)
    a3, b3, c3 = calculate_abc(C, D, A, d)
    a4, b4, c4 = calculate_abc(D, A, B, d)
    B1 = get_intersection(a1, b1, c1, a2, b2, c2)
    C1 = get_intersection(a2, b2, c2, a3, b3, c3)
    D1 = get_intersection(a3, b3, c3, a4, b4, c4)
    A1 = get_intersection(a4, b4, c4, a1, b1, c1)
    
    res = [A1, B1, C1, D1]
    res = [((p[0]), (p[1])) for p in res]

    return res


