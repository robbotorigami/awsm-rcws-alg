from forwardmodel import generator
from displaytools import rcws
from wftools.datatypes import zernikie_wfe, optical_setup
from imagetools import preprocess, featureextraction

dispwin = rcws.window(False)

wfe = zernikie_wfe([0, 0.07])

opt = optical_setup(5.86e-6, 0.6096, 0.55e-6, 1e-3, 0.05)

pre_im, pos_im = generator.generate_images(wfe, opt)
dispwin.display_prepost(pre_im, pos_im)
pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 13)
pre_im, pos_im = preprocess.normalize(pre_im, pos_im)
com1, com2 = featureextraction.find_com(pre_im, pos_im)
print(com2[0] - com1[0])
dispwin.display_centroid(pre_im, pos_im, (com1, com2))
ttmag, pre_im, pos_im = featureextraction.parse_tip_tilt(pre_im, pos_im, opt)
print(ttmag)



#dispwin.display_prepost(pre_im, pos_im, cross_section=True)

"""
for i in range(11):
    for defocus in [0.1e-3, 0.720e-3, 1e-3, 10e-3]:
        zern = [0] * 11
        zern[i] = 0.07
        wfe = zernikie_wfe(zern)
        pre_im, pos_im = generator.generate_images(wfe, defocus)
        pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 11)
        dispwin.display_prepost(pre_im, pos_im, cross_section = True)
"""
