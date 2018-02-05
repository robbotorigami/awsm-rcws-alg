from forwardmodel import generator
from displaytools import rcws

dispwin = rcws.window()

for i in range(11):
    for defocus in [0.1e-3, 0.720e-3, 1e-3, 10e-3]:
        zern = [0] * 11
        zern[i] = 0.07
        wfe = generator.zernikies(zern)
        pre_im, pos_im = generator.generate_images(wfe, defocus)
        dispwin.display_prepost(pre_im, pos_im)