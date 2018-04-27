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

def loadTrialImage(n):
    fname = "resources/Tilt Tip Stage/2018-03-29/Test 1/16_40_52Z_Test 1_{:04d}.png".format(n)

    im = imageio.imread(fname)
    #im = np.sum(im, axis=2)
    return im

def loadSetImages(s):
    return loadTrialImage(s[0])

def loadDataSets():
    sets = []
    with open('resources/Tilt Tip Stage/TipTilts.csv', 'r') as f:
        creader = csv.reader(f)
        for line in creader:
            sets.append((int(line[0]), float(line[1]),float(line[2])))

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

        print(cm1, cm2)
        outputdata.append((s[0], s[1], s[2], cm1[0], cm1[1]))

        #dispwin.display_prepost(pre_im, pos_im, False, coms = (cm1, cm2))

    with open('plots/TipTilts.csv', 'w') as f:
        cwriter = csv.writer(f)
        for line in outputdata:
            cwriter.writerow([str(l) for l in line])

    xs = [r[3] for r in outputdata]
    tilts = [r[1] for r in outputdata]

    plt.ioff()
    plt.figure()
    plt.plot(tilts, xs, '-ko')
    plt.show()
    quit()
