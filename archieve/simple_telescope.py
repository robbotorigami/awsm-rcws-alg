import proper
import matplotlib.pyplot as plt
from matplotlib import cm

def simple_telescope(wavelength, gridsize):
    diam = 1.0
    focal_ratio = 15.0
    focal_length = diam * focal_ratio
    beam_ratio = 0.5

    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)

    proper.prop_circular_aperture(wfo, diam/2)
    proper.prop_zernikes(wfo, [5], [1e-6])
    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo, focal_length*0.98)

    proper.prop_propagate(wfo, focal_length)

    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)

if __name__ == "__main__":
    psf, sampling = proper.prop_run( 'simple_telescope', 0.5, 512)
    plt.imshow(psf, cmap=cm.gray)
    plt.show()

