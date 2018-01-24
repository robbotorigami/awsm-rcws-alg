import proper
import matplotlib.pyplot as plt
import numpy as np

#define a prescription for proper for the prefocal image
def prefocal_image(wavelength, gridsize, PASSVAL):
    diam = PASSVAL['diam']
    focal_length = PASSVAL['focal_length']
    beam_ratio = PASSVAL['beam_ratio']

    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)

    proper.prop_circular_aperture(wfo, diam/2)
    proper.prop_define_entrance(wfo)
    proper.prop_zernikes(wfo, [i+1 for i in range(len(PASSVAL['ZERN']))], PASSVAL['ZERN'])
    proper.prop_lens(wfo, focal_length)

    proper.prop_propagate(wfo, focal_length - PASSVAL['DEFOCUS'], TO_PLANE=False)
    wfo_s = proper.prop_get_amplitude(wfo)
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
