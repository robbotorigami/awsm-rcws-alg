import cv2
import numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.patches as mpatches
from scipy import interpolate
from scipy.ndimage import morphology


#Return a masked binary image of the perimeter
def create_masks(im1, im2):
    max1 = np.max(im1)
    max2 = np.max(im2)
    im1 = (im1 > max1*0.5).astype(int)
    im2 = (im2 > max2*0.5).astype(int)
    return im1, im2, im1*im2

def extract_laplacians(preim, posim, masks, opt):
    diff = (preim - posim)/(preim+posim)
    laplacian = opt.focal_length*(opt.focal_length - opt.defocus)/(opt.defocus) * diff
    laplacian += 1e-6
    laplacian *= masks[2]
    laplacian *= 1e-4
    return laplacian

def rt_to_xy(rho, theta, shape):
    x = int(rho * np.cos(theta) + shape[0]/2)
    y = int(rho * np.sin(theta) + shape[1]/2)
    return x, y

def extract_normals(preim, posim, masks, opt):
    diff = (preim - posim)/(preim+posim)
    normalsx = 0*preim
    normalsy = 0*posim

    height, width = masks[0].shape

    #For each row in the image, comput the x derivative on the left and right
    for y in range(height):
        #Find the width on the left
        dxl = masks[0][y,:width//2] - masks[1][y,:width//2]
        #Find the width on the right
        dxr = masks[0][y, width//2:] - masks[1][y,width//2:]

        #Find the pixel to store the normal on the left and right
        xl = np.argmax(masks[2][y,:] > 0)
        xr = width - np.argmax(masks[2][y,::-1] > 0)

        #Insert the value into the x normal matrix
        normalsx[y, xl] = dxl
        normalsx[y, xr] = dxr

    # For each column in the image, compute the y derivative on the top and bottom
    for x in range(width):
        # Find the width on the left
        dyb = masks[0][height//2:, x] - masks[1][height//2:, x]
        # Find the width on the right
        dyt = masks[0][:height//2, x] - masks[1][height//2:, x]

        # Find the pixel to store the normal on the left and right
        yt = np.argmax(masks[2][:, x] > 0)
        yb = height - np.argmax(masks[2][::-1, x] > 0)

        # Insert the value into the x normal matrix
        normalsx[yb, x] = dyb
        normalsx[yt, x] = dyt


    #For a number of points around the spot
    pointlist = []
    for theta in np.linspace(0, 2*np.pi, 9):
        #1. Determine the first pixel in the direction of theta
        #   outside the combined mask
        rho = 0
        x, y = rt_to_xy(rho, theta, diff.shape)
        while masks[2][y, x] > 0.5:
            rho += 1
            x, y = rt_to_xy(rho, theta, diff.shape)
        #Now we have rho and theta of the first pixel outside of the mask
        #We want to keep the pixels location
        pix = (x, y)
        #Now, find the integral of the rest of the pixels in that direction
        x, y = rt_to_xy(rho, theta, diff.shape)
        accum = 0
        while x >= 0 and y >= 0 and x < diff.shape[0] and y < diff.shape[1]:
            accum += diff[y,x]
            rho += 1
            x, y = rt_to_xy(rho, theta, diff.shape)
        #print(pix, accum)
        pointlist.append((pix[0],pix[1], accum))
    #print(pointlist)
    xs, ys, values = zip(*pointlist)
    #print(xs, ys, values)
    sampler = interpolate.interp2d(xs, ys, values)
    xcoords = range(diff.shape[0])
    ycoords = range(diff.shape[1])
    normals = sampler(xcoords, ycoords)
    normals *= (1-masks[2])
    normals *= morphology.binary_dilation(masks[2])

    #Perform scaling on the array
    normals *= opt.focal_length * (opt.focal_length - opt.defocus) / (opt.defocus) * opt.pixel_size
    return normals

#find the center of mass for the two images
def find_com(im1, im2):
    c1, c2 = [0, 0], [0, 0]
    s1, s2 = 0,0
    res = len(im1)
    for x in range(res):
        for y in range(res):
            c1[0] += x*im1[y][x]
            c1[1] += y*im1[y][x]
            c2[0] += x*im2[y][x]
            c2[1] += y*im2[y][x]
            s1 += im1[y][x]
            s2 += im2[y][x]
    return [c/s1 for c in c1], [c/s2 for c in c2]

#Shift the images according to the provided values
def shift_images(im1, im2, cd):
    num_rows, num_cols = im1.shape[:2]

    translation_matrix = np.float32([[1, 0, -cd[0]/2], [0, 1, -cd[1]/2]])
    im1 = cv2.warpAffine(im1, translation_matrix, (num_cols, num_rows))
    translation_matrix = np.float32([[1, 0, cd[0]/2], [0, 1, cd[1]/2]])
    im2 = cv2.warpAffine(im2, translation_matrix, (num_cols, num_rows))
    return im1, im2

#returns the tip and tilt, and the images with tip tilt removed
def parse_tip_tilt(preim, postim, opt):
    c1, c2 = find_com(preim, postim)
    cd = np.array([e1 - e2 for e1, e2 in zip(c1, c2)])
    preim, posim = shift_images(preim, postim, cd)

    #calculate rate (in pix/m)
    dzdr = cd * opt.defocus/(2*opt.focal_length*(opt.focal_length - opt.defocus))

    #Convert to m/m
    dzdr *= opt.pixel_size

    #Convert to waves/5cm
    dzdr /= opt.wavelength
    dzdr *= opt.aperature
    #Convert from ptp to rms
    mag = dzdr/4

    return mag, preim, posim

if __name__ == "__main__":
    #Go through a range of defocuses and identify tip and tilt

    zern = [-0.2868445,
            0,
            0.08498557,
            -0.16551221,
            0,
            0.16330520,
            0.03002613,
            0,
            0.00034476,
            0,
            -0.0008443]
    plt.ion()
    plt.show()

    dfl = [0.1e-3, 0.2e-3, 0.4e-3, 0.8e-3, 1.6e-3, 3.2e-3, 6.4e-3]
    dfl = np.arange(0.4e-3, 1.0e-3, 0.05e-3)
    error = []

    preim, postim = gen.generate_images(zern, defocus= 1.6e-3)
    preim, postim = pp.normalize(preim, postim)
    preim, postim = pp.blur_images(preim, postim)
    cv_blob_detect(preim, postim)
    exit()
    for defocus in dfl:
        preim, postim = gen.generate_images(zern, defocus = defocus)
        preim, postim = pp.normalize(preim, postim)
        preim, postim = pp.blur_images(preim, postim)
        dzdr, preim, postim = parse_tip_tilt(preim, postim, defocus)
        error.append(abs(dzdr - zern[1:3])*100)

        diff = preim - postim
        plt.subplot(1,3,1)
        plt.title("Prefocal")
        plt.imshow(preim, cmap=cm.gray)
        plt.subplot(1,3,2)
        plt.title("Postfocal")
        plt.imshow(postim, cmap=cm.gray)
        plt.subplot(1,3,3)
        plt.title("Difference")
        plt.imshow(diff, cmap=cm.gray)
        plt.draw()
        plt.pause(0.001)
        #plt.savefig("plots/" + 'OurZernikies' + '_{}um'.format(int(defocus*1e6)) + ".png")



    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Defocus Distance (m)')
    ax1.plot(dfl, [e[0] for e in error], 'r')
    ax1.plot(dfl, [e[1] for e in error], 'b')
    ax1.plot(dfl, [np.sqrt(e[0]**2 + e[1]**2) for e in error], 'g')
    #ax1.set_ylim((0,100))
    ax1.set_ylabel('Error (%)', color='r')
    ax1.tick_params('y', colors='r')

    red_patch = mpatches.Patch(color='red', label='Z2')
    blue_patch = mpatches.Patch(color='blue', label='Z3')
    green_patch = mpatches.Patch(color='green', label='Combined')
    plt.legend(handles=[red_patch, blue_patch, green_patch])

    fig.tight_layout()
    plt.show()
    plt.pause(10)
    plt.savefig("plots/" + 'OurZernikiesError.png')

    #Commented out section will be used later for higher order aberrations

    """
    
    
    defocus = 720e-6

    input = np.arange(0.04, 0.40, 0.01)
    output = []
    error = []
    zern = [-0.2868445,
            0,
            0.08498557,
            -0.16551221,
            0,
            0.16330520,
            0.03002613,
            0,
            0.00034476,
            0,
            -0.0008443]

    preim, postim = gen.generate_images(zern, defocus = defocus)
    preim, postim = pp.baseline_process(preim, postim)
    dzdr, preim, postim = parse_tip_tilt(preim, postim, defocus)

    coef = dzdr[1]
    print(coef)

    """
    """

    for i in input:
        preim, postim = gen.generate_images([0,i], defocus = defocus)
        preim, postim = pp.baseline_process(preim, postim)
        cd, preim, postim = parse_tip_tilt(preim, postim)

        coef = dzdr[0]
        output.append(coef)
        error.append(abs((i-coef)/i)*100)

    #plt.plot(input.tolist(), output)
    #plt.plot(input.tolist(), error)
    #plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(input, output, 'b')
    ax1.set_xlabel('Z2 (lambda RMS)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Z2 (lambda RMS)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(input, error, 'r.')
    ax2.set_ylim((0,100))
    ax2.set_ylabel('Error (%)', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()


    #shrink mask by 10 pixels
    #mask = scipy.ndimage.binary_erosion(mask, iterations = 5).astype(mask.dtype)
    """