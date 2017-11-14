import zernike
import time
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import pytest

#@pytest.mark.skip("not really pass fail")
def test_profiling():
    random.seed(1235434)
    rand_coef = [random.random() for i in range(15)]
    start = time.time()
    image = zernike.generate_wavefront(rand_coef, 200, 200)
    end_zern = time.time()
    inv_coef = zernike.map_to_zernike(image)
    end_inv = time.time()

    print('Generating image took {}, computing ZPC took {}'.format(end_zern - start, end_inv-end_zern))

def test_accuracy(show_res = True):
    avg = 5
    res = 200
    max_error = 0.05
    random.seed(125434)
    if show_res:
        fig = plt.figure()
    for i in range(avg):
        rand_coef = [random.random()-0.5 for i in range(15)]
        image = zernike.generate_wavefront(rand_coef, res, res)
        inv_coef = zernike.map_to_zernike(image)
        if show_res:
            plt.subplot(3, avg, i+1)
            plt.imshow(image, cmap=cm.coolwarm)
            plt.title('Trial {} : start'.format(i))

            plt.subplot(3, avg, avg+i+1)
            recon = zernike.generate_wavefront(inv_coef, res, res)
            plt.imshow(recon, cmap=cm.coolwarm)
            plt.title('Trial {} : reconstruct'.format(i))

            plt.subplot(3, avg, 2*avg+i+1)
            plt.imshow([[o-r for o, r in zip(rowi, rowr)] for rowi, rowr in zip(image, recon)])
            plt.title('Trial {} : error'.format(i))
        norm = [math.fabs(r - i) for r, i in zip(rand_coef, inv_coef)]
        assert(len([n for n in norm if n > max_error]) == 0)
    plt.show()