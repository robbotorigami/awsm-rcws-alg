import proper

def simple_telescope(wavelength, gridsize):
    d_objective = 0.060
    fl_objective = 15.0 * d_objective
    fl_eyepiece = 0.021
    fl_eye = 0.022
    beam_ratio = 0.5

    wfo = proper.propbegin(d_objective, wavelength, gridsize, beam_ratio)

    proper.prop_circular_aperture(wfo, d_objective / 2)
    proper.prop_define_entrance(wfo)

    proper.prop_lens(wfo, fl_objective, "objective")

    proper.prop_propagate(wfo, fl_objective + fl_eyepiece, "eyepiece")
    proper.prop_lens(wfo, fl_eyepiece, "eyepiece")

    exit_pupil_distance = fl_eyepiece / (1 - fl_eyepiece / (fl_objective + fl_eyepiece))
    proper.prop_propagate(wfo, exit_pupil_distance, "exit pupil at eye lens:")
    proper.prop_lenx(wfo, fl_eye, "eye")

    proper.prop_propagate(wfo, fl_eye, "retina")

    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)

psf, sampling = proper.prop_run( 'simple_telescope', 0.5, 512)
