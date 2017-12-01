import prefocal_image
import matplotlib.pyplot as plt, matplotlib.cm as cm
import proper
import math

def generate_images(zernikies, defocus = 0.01):
    wavelength = 0.55 #um
    pix_size = 5.86e-6
    alpha = 0.1396
    height = defocus*math.sin(alpha)
    beam_ratio = 0.5
    npix = height/beam_ratio/pix_size #number of pixels that the bean will cover
    gridsize = int(2**(int(math.log(npix, 2))+1)) #npix to next power of 2
    setting = {'ZERN': zernikies, 'DEFOCUS': defocus,
               'diam': 0.05, 'focal_length': 0.625, 'beam_ratio': beam_ratio, 'det_size' : 0.1}

    pre_im, sampling = proper.prop_run( 'prefocal_image', 0.55, gridsize, PASSVALUE=setting)
    pos_im, sampling = proper.prop_run( 'postfocal_image', 0.55, gridsize, PASSVALUE=setting)
    #invert postfocal image
    pos_im = [[pos_im[gridsize - y-1][gridsize - x-1] for x in range(gridsize)] for y in range(gridsize)]

    return pre_im, pos_im


if __name__ == "__main__":
    pre_im, pos_im = generate_images([0,0,-5e-6,0,0,0,5e-6,0,0,0,0], 0.01)
    wavelength = 0.55 #um
    pix_size = 5.86e-6
    cmos_size = 7.03e-3 #m
    defocus = 0.01
    alpha = 0.1396
    height = defocus*math.sin(alpha)
    beam_ratio = 0.5
    npix = height/beam_ratio/pix_size
    gridsize = int(2**(int(math.log(npix, 2))+1))
    print(gridsize)
    setting = {'ZERN': [0,0,-wavelength*10*1e-6,0,0,0,wavelength*1*1e-6,0,0,0,0], 'DEFOCUS': defocus,
               'diam': 0.05, 'focal_length': 0.625, 'beam_ratio': beam_ratio, 'det_size' : 0.1}
    """
    for i in range(10):
        setting['DEFOCUS'] = i * 0.00001
        pre_im, sampling = proper.prop_run( 'prefocal_image', 0.55, gridsize, PASSVALUE=setting)
        plt.subplot(4,3,i+1)
        plt.imshow(pre_im, cmap=cm.gray)
    plt.show()
    """

    pre_im, sampling = proper.prop_run( 'prefocal_image', 0.55, gridsize, PASSVALUE=setting)
    pos_im, sampling = proper.prop_run( 'postfocal_image', 0.55, gridsize, PASSVALUE=setting)
    #invert postfocal image
    pos_im = [[pos_im[gridsize - y-1][gridsize - x-1] for x in range(gridsize)] for y in range(gridsize)]
    plt.subplot(1,3,1)
    plt.imshow(pre_im, cmap=cm.gray)
    plt.subplot(1,3,2)
    plt.imshow(pos_im, cmap=cm.gray)
    plt.subplot(1,3,3)
    plt.imshow([[pos - pre for pos, pre in zip(por, prr)] for por, prr in zip(pos_im, pre_im)], cmap=cm.gray)

    plt.show()