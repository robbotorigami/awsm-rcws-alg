import proper
import numpy as np
import cv2
import os, sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from wftools.datatypes import zernike_wfe, optical_setup

#uses proper to simulate the expected images
def generate_images(zern, opt):
    pix_size = opt.pixel_size
    height = opt.sensor_height()
    beam_ratio = 0.5
    npix = int(height/beam_ratio/pix_size) #number of pixels that the grid should span
    gridsize = 512
    setting = {
                'ZERN': zern.at_wavelength(opt.wavelength),
                'DEFOCUS': opt.defocus,
                'diam': opt.aperature,
                'focal_length': opt.focal_length,
                'beam_ratio': beam_ratio
               }

    #add the prescriptions to the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    #run the optical simulation
    pre_im, sampling = proper.prop_run('prefocal_image', opt.wavelength_as_um(), gridsize, PASSVALUE=setting, QUIET = True)
    pos_im, sampling = proper.prop_run('postfocal_image', opt.wavelength_as_um(), gridsize, PASSVALUE=setting, QUIET = True)

    #invert postfocal image
    pos_im = [[pos_im[gridsize - y-1][gridsize - x-1] for x in range(gridsize)] for y in range(gridsize)]

    pre_im = cv2.resize(np.array(pre_im), (npix, npix))
    pos_im = cv2.resize(np.array(pos_im), (npix, npix))

    return pre_im, pos_im
