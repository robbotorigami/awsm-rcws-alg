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

def loadTrialImage(t, n):
    if t == 1:
        fname = "resources/Linear Traverse/2018-03-28/Test {}/22_21_04Z_QHY_{:04d}.png".format(t, n)

    if t == 2:
        fname = "resources/Linear Traverse/2018-03-28/Test {}/22_36_34Z_Test {}_{:04d}.png".format(t, t, n)

    im = imageio.imread(fname)
    #im = np.sum(im, axis=2)
    return im

def loadSetImages(s):
    return loadTrialImage(s[0], s[1])

def loadDataSets():
    sets = []
    with open('resources/Linear Traverse/Positions.csv', 'r') as f:
        creader = csv.reader(f)
        for line in creader:
            sets.append((1, int(line[0]), float(line[1])))
    with open('resources/Linear Traverse/Positions.csv', 'r') as f:
        creader = csv.reader(f)
        for line in creader:
            sets.append((2, int(line[0]), float(line[1])))

    return sets


if __name__ == '__main__':
    sets = loadDataSets()
    print(sets)
    dispwin = rcws.window(True)

    outputdata = []

    for s in sets:
        pre_im = loadSetImages(s)
        pos_im = np.ones_like(pre_im)
        cm1, cm2 = featureextraction.find_com(pre_im, pos_im)

        crop, _ = preprocess.crop(pre_im, pos_im*0)
        width = crop.shape[1]
        height = crop.shape[0]
        angle = np.arcsin(height/width)*180 / np.pi

        print(cm1, cm2)
        outputdata.append((s[0], s[1], s[2], cm1[0], cm1[1], angle))

        #dispwin.display_prepost(pre_im, pos_im, False, coms = (cm1, cm2))

    with open('plots/Linear.csv', 'w') as f:
        cwriter = csv.writer(f)
        cwriter.writerow(['Test', 'Image', 'Displacement', 'x center', 'y center', 'angle'])
        for line in outputdata:
            cwriter.writerow([str(l) for l in line])

    xs = [r[3] for r in outputdata]
    tilts = [r[1] for r in outputdata]
    angles = [r[5] for r in outputdata]

    plt.ioff()
    plt.figure()
    plt.plot(tilts, xs, '-ko')
    plt.show()
    plt.figure()
    plt.plot(angles)
    plt.show()
    quit()
