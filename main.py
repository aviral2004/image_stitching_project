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

def get_bounding_box(image, H):
    # get all corners
    corners = np.array([
        [0, 0, 1],
        [image.shape[1], 0, 1],
        [0, image.shape[0], 1],
        [image.shape[1], image.shape[0], 1],
    ]).T

    # transform corners
    corners = H @ corners

    # normalize: x/w, y/w
    corners = corners / corners[2]

    # get bounding box
    min_x = int(np.floor(np.min(corners[0])))
    max_x = int(np.ceil(np.max(corners[0])))
    min_y = int(np.floor(np.min(corners[1])))
    max_y = int(np.ceil(np.max(corners[1])))

    return min_x, min_y, max_x, max_y

def get_translation_mat(delta_x, delta_y):
    return np.array(
        [[1, 0, delta_x],
         [0, 1, delta_y],
         [0, 0, 1]
         ],
        dtype=np.float32
    )

def get_canvas_box(boxes):
    # get the bounding box of the corners
    tl_corners = np.array([(box[0], box[1]) for box in boxes])
    br_corners = np.array([(box[2], box[3]) for box in boxes])

    min_x = int(np.min(tl_corners[:, 0]))
    min_y = int(np.min(tl_corners[:, 1]))

    max_x = int(np.max(br_corners[:, 0]))
    max_y = int(np.max(br_corners[:, 1]))

    return (min_x, min_y), (max_x - min_x, max_y - min_y)

def get_warped_images(images, Hs):
    # get bounding boxes
    boxes = [get_bounding_box(image, H) for image, H in zip(images, Hs)]
    print(boxes)

    # get canvas box
    global_tl_corner, global_dim = get_canvas_box(boxes)
    print(global_tl_corner, global_dim)

    # get translation matrices
    T = get_translation_mat(-global_tl_corner[0], -global_tl_corner[1])

    # transform images
    transformed_images = [cv2.warpPerspective(image, T @ H, global_dim) for image, H in zip(images, Hs)]

    return transformed_images

def stitch_images(images):
    # do and mask and blend
    masks = [np.any(im != [0, 0, 0], axis=2) for im in images]
    common = np.logical_and(masks[0], masks[1])
    common_image = cv2.addWeighted(images[0], 0.5, images[1], 0.5, 0)
    final_image = images[0] + images[1]
    final_image[common] = common_image[common]

    return final_image