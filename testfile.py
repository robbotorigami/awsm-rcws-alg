from forwardmodel import generator
from displaytools import rcws
from wftools.datatypes import zernike_wfe, optical_setup
from imagetools import preprocess, featureextraction
from algorithm import finitedifferences
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt, matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.ndimage import morphology



shape = (100, 100)
image = [[finitedifferences.xy_to_rt(i,j,shape)[1] for i in range(100)] for j in range(100)]

#plt.imshow(image, cmap=cm.gray)
#plt.show()



dispwin = rcws.window(False)

wfe = zernike_wfe([0, 0.07])
opt = optical_setup(5.86e-6, 0.6096, 0.55e-6, 1e-3, 0.05)
pre_im, pos_im = generator.generate_images(wfe, opt)
pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 21)
pre_im, pos_im = preprocess.normalize(pre_im, pos_im)
#dispwin.display_prepost(pre_im, pos_im, cross_section = True)
pre_mask, pos_mask, comb_mask = masks = featureextraction.create_masks(pre_im, pos_im)
#dispwin.display_prepost_masks(pre_im, pos_im, masks)
laplacian = featureextraction.extract_laplacians(pre_im, pos_im, masks, opt)
normals = featureextraction.extract_normals(pre_im, pos_im, masks, opt)
dispwin.display_features(pre_im, pos_im, laplacian, normals)
wavefront = finitedifferences.solve_wavefront(laplacian, normals)

from scipy.ndimage import filters
plt.subplot(1,3,1)
plt.imshow(filters.laplace(wavefront)*comb_mask, cmap = cm.gray)

rad_grad = cv2.resize(np.array(wavefront), (wavefront.shape[0]+2,wavefront.shape[1]+2))
rad_grad = rad_grad - cv2.copyMakeBorder(wavefront, top=1, bottom=1, left=1, right=1, borderType= cv2.BORDER_CONSTANT)
rad_grad = rad_grad[1:-1, 1:-1]
rad_grad *= 1-comb_mask
rad_grad *= morphology.binary_dilation(masks[2])

plt.subplot(1,3,2)
plt.imshow(-rad_grad, cmap = cm.gray)
plt.subplot(1,3,3)
plt.imshow(wavefront, cmap = cm.gray)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = range(wavefront.shape[1])
y = range(wavefront.shape[0])
X, Y = np.meshgrid(x,y)
ax.plot_surface(X, Y, wavefront, cmap=cm.coolwarm)
plt.show()


"""
wfe = zernike_wfe([0, 0.07])

opt = optical_setup(5.86e-6, 0.6096, 0.55e-6, 1e-3, 0.05)

pre_im, pos_im = generator.generate_images(wfe, opt)
#dispwin.display_prepost(pre_im, pos_im, cross_section = True)
pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 1)
pre_im, pos_im = preprocess.normalize(pre_im, pos_im)
dispwin.display_prepost(pre_im, pos_im, cross_section = True)
com1, com2 = featureextraction.find_com(pre_im, pos_im)
print(com2[0] - com1[0])
#dispwin.display_centroid(pre_im, pos_im, (com1, com2))
ttmag, pre_im, pos_im = featureextraction.parse_tip_tilt(pre_im, pos_im, opt)
print(ttmag)


#dispwin.display_prepost(pre_im, pos_im, cross_section=True)

for i in range(11):
    for defocus in [0.1e-3, 0.720e-3, 1e-3, 10e-3]:
        opt.defocus = defocus
        zern = [0] * 11
        zern[i] = 0.07
        wfe = zernikie_wfe(zern)
        pre_im, pos_im = generator.generate_images(wfe, opt)
        pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 11)
        dispwin.display_prepost(pre_im, pos_im, cross_section = True)

"""
