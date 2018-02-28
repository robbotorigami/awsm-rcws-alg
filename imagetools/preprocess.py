import cv2
import numpy as np

## Provides standard methods to pre process the image

def normalize(im1, im2):
    norm1 = im1/im1.max()
    norm2 = im2/im2.max()
    return norm1, norm2

def blur_images(im1, im2, bluramount = 5, gaussian = True):
    if gaussian:
        blur1 = cv2.GaussianBlur(im1, (bluramount, bluramount), 0)
        blur2 = cv2.GaussianBlur(im2, (bluramount, bluramount), 0)
    else:
        blur1 = cv2.blur(im1, (bluramount, bluramount), 0)
        blur2 = cv2.blur(im2, (bluramount, bluramount), 0)
    return blur1, blur2

def inside_mask(im1, im2, threshold = 1):
    mask1 = (im1 > threshold)
    mask2 = (im2 > threshold)
    return mask1, mask2, np.logical_and(mask1, mask2)

def baseline_process(preim, postim):
    preim, postim = normalize(preim, postim)
    preim, postim = blur_images(preim, postim)

    return preim, postim
