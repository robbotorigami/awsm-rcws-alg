import proper

def postfocal_image(wavelength, gridsize, PASSVAL):
    diam = PASSVAL['diam']
    focal_length = PASSVAL['focal_length']
    beam_ratio = PASSVAL['beam_ratio']
    det_size = PASSVAL['det_size']

    wfo = proper.prop_begin(diam*2, wavelength, gridsize, beam_ratio)

    proper.prop_circular_aperture(wfo, diam)
    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo, focal_length)
    proper.prop_zernikes(wfo, [i+1 for i in range(len(PASSVAL['ZERN']))], PASSVAL['ZERN'])

    proper.prop_propagate(wfo, focal_length + PASSVAL['DEFOCUS'])

    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
