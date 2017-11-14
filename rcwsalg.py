import math
import zernike
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def xytort(x, y):
    rho = math.sqrt((x**2 + y**2))
    theta = math.atan2(y, x)
    return rho, theta

def rttoxy(rho, theta):
    x = int(math.round(rho*math.cos(theta)))
    y = int(math.round(rho*math.sin(theta)))
    return x,y

def computeImages(wavefront, useDiags = False, givederivs = False):
    dx, dy = tuple(np.gradient(wavefront.image))
    d2x, d2y = np.gradient(dx), np.gradient(dy)

    laplacian = d2x + d2y

    dn = np.sqrt(np.power(dx, 2) + np.power(dy, 2))

    secder = [[0 for i in range(x_res)] for j in range(y_res)]
    firstder = [[0 for i in range(x_res)] for j in range(y_res)]
    sdx = [[0 for i in range(x_res)] for j in range(y_res)]
    sdy = [[0 for i in range(x_res)] for j in range(y_res)]
    #note I am not implementing the first derivative stuff yet!
    for y in range(1, y_res - 1):
        for x in range(1, x_res - 1):
            if xytort(X[y][x], Y[y][x])[0] > 0.97:
                secder[y][x] = 0
                continue
            xdelt = wavefront[y][x+1] + wavefront[y][x-1] - 2*wavefront[y][x]
            ydelt = wavefront[y+1][x] + wavefront[y-1][x] - 2*wavefront[y][x]
            lap = xdelt + ydelt
            secder[y][x] = lap
            firstder[y][x] = wavefront[y][x+1] - wavefront[y][x] + wavefront[y+1][x] - wavefront[y][x]
            sdx[y][x] = xdelt
            sdy[y][x] = ydelt
            if useDiags:
                cross1 = wavefront[y+1][x+1] + wavefront[y-1][x-1] - 2*wavefront[y][x]
                cross2 = wavefront[y+1][x-1] + wavefront[y-1][x+1] - 2*wavefront[y][x]
                secder[y][x] += cross1 + cross2
    if givederivs:
        return sdx, sdy
    img1 = [[d for d in dr] for dr in secder]
    img2 = [[-d for d in dr] for dr in secder]

    return img1, img2

if __name__ == "__main__":
    coeff = [0,0,0,0,0,0.1]
    wavefront = zernike.generate_wavefront(coeff, 200, 200, 100)
    im1, im2 = computeImages(wavefront)

    plt.subplot(1, 3, 1)
    plt.imshow(wavefront, cmap=cm.coolwarm)
    plt.subplot(1, 3, 2)
    plt.imshow(im1, cmap=cm.Greys)
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap=cm.Greys)
    plt.show()
