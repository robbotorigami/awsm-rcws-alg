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

import csv
import imageio
import copy

def loadTrialImage(n):
    fname = "resources/dataset1/2018-04-13/Test 1/21_15_05Z_Test 1_{:04d}.png".format(n)

    im = imageio.imread(fname)
    im = np.sum(im, axis=2)
    return im

def loadSetImages(set):
    pre_im = loadTrialImage(set[0])
    pos_im = loadTrialImage(set[1])
    return pre_im, pos_im

def loadTrial1DataSets():
    sets = []
    with open('resources/dataset1/tests.csv', 'r') as f:
        creader = csv.reader(f)
        for line in creader:
            sets.append(float(line[1]))

    sets = [(i+1, s - sets[0]) for i,s in enumerate(sets)]
    sets.sort(key = lambda x: x[1])
    sets = [(s1[0], s2[0], s2[1]) for s1, s2 in zip(sets, reversed(sets))][:len(sets)//2]

    return sets

def getWFE(dataset, showplots = False):
    #Fetch the images to be used
    pre_im, pos_im = loadSetImages(dataset)

    #Setup the assumed optical system
    opt = optical_setup(5.86e-6, 0.6096, 0.55e-6, dataset[2], 0.05)

    #Perfom preprocessing and cropping tasks
    pre_im, pos_im = preprocess.crop(pre_im, pos_im)
    if showplots:
        plt.subplot(1,4,1)
        plt.imshow(pre_im, cmap = cm.gray, interpolation='none')
        plt.subplot(1,4,2)
        plt.imshow(pos_im, cmap = cm.gray, interpolation='none')
    #pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 21)
    pre_im, pos_im = preprocess.normalize(pre_im, pos_im)

    #Extract the relavent features from the image
    coms, pre_im, pos_im = featureextraction.parse_tip_tilt(pre_im, pos_im, opt)
    pre_mask, pos_mask, comb_mask = masks = featureextraction.create_masks(pre_im, pos_im)
    laplacian = featureextraction.extract_laplacians(pre_im, pos_im, masks, opt)
    normals = featureextraction.extract_normals(pre_im, pos_im, masks, opt)

    #Reconstruct the wavefront from the extracted features
    wavefront = finitedifferences.solve_wavefront(laplacian, normals)

    if showplots:
        plt.subplot(1,4,3)
        plt.imshow(wavefront, cmap = cm.gray, interpolation='none')

    recovered = zernike_wfe.from_image(wavefront)

    #recovered.coef[1] -= coms[0]
    #recovered.coef[2] -= coms[1]

    wfe2 = copy.deepcopy(recovered)

    wfe2.coef[1] = 0
    wfe2.coef[2] = 0

    img = wfe2.as_image(512, 512, 1, True)
    print(dataset[2])
    if showplots:
        plt.subplot(1,4,4)
        plt.imshow(img, cmap = cm.hot, interpolation='none')
        plt.show()

    return recovered



if __name__ == '__main__':
    sets = loadTrial1DataSets()
    print(sets)

    dispwin = rcws.window(False)

    results = []

    for dataset in sets[1:4]:
        wfe = getWFE(dataset, True)
        results.append((dataset[2], wfe))


    tilts = [r[1].coef[1] for r in results]
    tilts2 = [r[1].coef[2] for r in results]
    astig = [r[1].coef[5] for r in results]
    defocuses = [r[0] for r in results]
    plt.figure()
    plt.xlim((0,6))
    plt.plot(defocuses, tilts, 'b')
    plt.plot(defocuses, tilts2, 'g')
    plt.plot(defocuses, astig, 'r')
    plt.show()
    quit()

    set = sets[2]
    pre_im, pos_im = loadSetImages(set)
    wfe = zernike_wfe([0, 0,0.07])
    opt = optical_setup(5.86e-6, 0.6096, 0.55e-6, 1e-3, 0.05)
    pre_im, pos_im = preprocess.crop(pre_im, pos_im)
    dispwin.display_prepost(pre_im, pos_im, False)
    pre_im, pos_im = preprocess.blur_images(pre_im, pos_im, 21)
    pre_im, pos_im = preprocess.normalize(pre_im, pos_im)
    coms, pre_im, pos_im = featureextraction.parse_tip_tilt(pre_im, pos_im, opt)
    print(coms)
    #dispwin.display_prepost(pre_im, pos_im, cross_section = True)
    pre_mask, pos_mask, comb_mask = masks = featureextraction.create_masks(pre_im, pos_im)
    laplacian = featureextraction.extract_laplacians(pre_im, pos_im, masks, opt)
    normals = featureextraction.extract_normals(pre_im, pos_im, masks, opt)
    #dispwin.display_features(pre_im, pos_im, laplacian, normals)


    wavefront = finitedifferences.solve_wavefront(laplacian, normals)

    # from scipy.ndimage import filters
    # plt.subplot(1,3,1)
    # plt.imshow(filters.laplace(wavefront)*comb_mask, cmap = cm.gray)
    #
    # rad_grad = cv2.resize(np.array(wavefront), (wavefront.shape[1]+2,wavefront.shape[0]+2))
    # rad_grad = rad_grad - cv2.copyMakeBorder(wavefront, top=1, bottom=1, left=1, right=1, borderType= cv2.BORDER_CONSTANT)
    # rad_grad = rad_grad[1:-1, 1:-1]
    # rad_grad *= 1-comb_mask
    # rad_grad *= morphology.binary_dilation(masks[2])
    #
    # plt.subplot(1,3,2)
    # plt.imshow(-rad_grad, cmap = cm.gray)
    # plt.subplot(1,3,3)
    # plt.imshow(wavefront, cmap = cm.gray)
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # x = range(wavefront.shape[1])
    # y = range(wavefront.shape[0])
    # X, Y = np.meshgrid(x,y)
    # ax.plot_surface(X, Y, wavefront, cmap=cm.coolwarm)
    # plt.show()

    recovered = wfe.from_image(wavefront)
    img = recovered.as_image(512, 512, 1, True)
    plt.imshow(img, cmap = cm.hot, interpolation='none')
    plt.title("Recovered Wavefront")
    plt.savefig("plots/wavefront2D.png")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img = recovered.as_image(16, 16, 1, False)
    x = range(img.shape[1])
    y = range(img.shape[0])
    X, Y = np.meshgrid(x,y)
    X, Y, Z = X[img != 0], Y[img != 0], img[img != 0]
    X, Y, Z = np.reshape(X, (X.size)), np.reshape(Y, (Y.size)), np.reshape(Z, (Z.size))

    ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)
    plt.show()


    recovered.coef[1] -= coms[0]
    recovered.coef[2] -= coms[1]

    coef = recovered.coef
    coef = np.array(coef)

    coef = coef / (np.max(np.abs(coef)))

    colors = []

    thresh = 0.1
    for i,val in enumerate(coef):
        if np.abs(val) > thresh:
            colors.append('r')
        else:
            colors.append('g')

    fig, ax = plt.subplots()
    ax.bar(range(len(coef)), coef, color = colors)
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticklabels([str(i) for i in range(7)])
    plt.savefig('plots/ZcoefAmp.png')
    plt.show()
    print(recovered.coef)
