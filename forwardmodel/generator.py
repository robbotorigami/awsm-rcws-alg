import matplotlib.pyplot as plt, matplotlib.cm as cm
import proper
import math
import numpy as np
import cv2
import os, sys

class zernikies:
    names = ['1_PISTON',
             '2_X tilt',
             '3_Y tilt',
             '4_Focus',
             '5_45DegAstigmatism',
             '6_0DegAstigmatism',
             '7_YComa',
             '8_XComa',
             '9_YTrefoil',
             '10_XTrefoil',
             '11_Spherical']

    def __init__(self, coef):
        self.coef = coef

    def at_wavelength(self, wavelength):
        return [z*0.55e-6 for z in self.coef]

#uses proper to simulate the expected images
def generate_images(zern, defocus = 0.01, wavelength = 0.55e-6):
    pix_size = 5.86e-6
    alpha = 0.1396
    height = defocus*math.sin(alpha)
    beam_ratio = 0.5
    npix = int(height/beam_ratio/pix_size) #number of pixels that the grid should span
    gridsize = 512
    setting = {
                'ZERN': zern.at_wavelength(wavelength),
                'DEFOCUS': defocus,
                'diam': 0.05,
                'focal_length': 0.6096,
                'beam_ratio': beam_ratio
               }

    #add the prescriptions to the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    #run the optical simulation
    pre_im, sampling = proper.prop_run('prefocal_image', wavelength*1e6, gridsize, PASSVALUE=setting, QUIET = True)
    pos_im, sampling = proper.prop_run('postfocal_image', wavelength*1e6, gridsize, PASSVALUE=setting, QUIET = True)

    #invert postfocal image
    pos_im = [[pos_im[gridsize - y-1][gridsize - x-1] for x in range(gridsize)] for y in range(gridsize)]

    pre_im = cv2.resize(np.array(pre_im), (npix, npix))
    pos_im = cv2.resize(np.array(pos_im), (npix, npix))

    return pre_im, pos_im