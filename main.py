import numpy as np
import numpy.linalg as la
import cv2
import sys

def gen_mat_A(p1, p2):
    x, y = p2
    _x, _y = p1
    return np.array(
        [[x, y, 1, 0, 0, 0, -x*_x, -y*_x],
         [0, 0, 0, x, y, 1, -x*_y, -y*_y],]
    )

def gen_mat_b(p1, p2):
    return np.array([p1[0], p1[1]])

def gen_full_mat_A(points1, points2):
    # vstack all individual A's
    A = np.vstack([gen_mat_A(points1[i], points2[i]) for i in range(len(points1))])
    return A

def gen_full_mat_B(points1, points2):
    # generate array of all q's
    b = np.hstack([gen_mat_b(points1[i], points2[i]) for i in range(len(points1))])
    return b

def computeH(points1, points2):
    A = gen_full_mat_A(points1, points2)
    B = gen_full_mat_B(points1, points2)
    mat = la.lstsq(A, B, rcond=None)[0]
    H = np.append(mat, 1).reshape(3, 3)
    return H