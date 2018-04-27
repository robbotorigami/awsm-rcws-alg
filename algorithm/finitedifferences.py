import numpy as np

def xy_to_rt(x, y, shape):
    y = y - shape[0]/2
    x = x - shape[1]/2
    rho = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(-y, x))%(2*np.pi)
    return rho, theta



def solve_wavefront(laplacian, normal):
    waveheight = laplacian.shape[0]
    wavewidth = laplacian.shape[1]

    #Build the indicies for every element in the wavefront
    #We will solve for any point that we have in either
    #the laplacian or the normal maps
    indicies_dict = {}

    idx = 0
    for y in range(0, waveheight):
        for x in range(0, wavewidth):
            if laplacian[y, x] != 0 or normal[y,x] != 0:
                indicies_dict[(x, y)] = idx
                idx += 1
    numelems = idx

    #Implement laplacian and normal features
    M = []
    f = []

    #Constrain the norm to be at 0
    r = [1 for i in range(numelems)]
    M.append(r)
    f.append(0)

    for y in range(0, waveheight):
        for x in range(0, wavewidth):
            #If there is a laplacian rule to set
            if laplacian[y, x] != 0:
                ######################################################
                #|----|----|----|
                #|    |  1 |    |
                #|----|----|----|
                #| 1  | -4 | 1  |
                #|----|----|----|
                #|    |  1 |    |
                #|----|----|----|
                #

                r = [0] * numelems
                r[indicies_dict[(x,y)]] = -4
                r[indicies_dict[(x-1, y)]] = 1
                r[indicies_dict[(x+1, y)]] = 1
                r[indicies_dict[(x, y-1)]] = 1
                r[indicies_dict[(x, y+1)]] = 1
                M.append(r)
                f.append(laplacian[y,x])
                #print(r, f)

            #If there is a normal rule to set
            if normal[y, x] != 0:
                ######################################################
                #|----|----|----|
                #|    | sin|    |
                #|----|----|----|
                #|-cos|  1 | cos|
                #|----|----|----|
                #|    |-sin|    |
                #|----|----|----|
                #
                #
                #|----|----|----|
                #|    | y-1|    |
                #|----|----|----|
                #| x-1|  1 | x+1|
                #|----|----|----|
                #|    | y+1|    |
                #|----|----|----|
                #The sins and cosines only used if < 0

                rho, theta = xy_to_rt(x, y, laplacian.shape)

                offsets = [(-1,0), (1,0), (0,-1), (0,1)]

                for offset in offsets:
                    xp = x + offset[0]
                    yp = y + offset[1]
                    if (xp, yp) in indicies_dict:
                        r = [0] * numelems
                        r[indicies_dict[(xp, yp)]] = -1
                        r[indicies_dict[(x,y)]]    =  1
                        M.append(r)
                        weight = (x-xp)* np.cos(theta) + (yp - y) * np.sin(theta)
                        f.append(weight*normal[y,x])


    #Solve Matrix equation
    #print(M)
    waveelems = np.linalg.lstsq(np.array(M), np.array(f))[0]
    #Insert into wavefront
    wavefront = np.zeros_like(laplacian)
    inv_indx = {v: k for k, v in indicies_dict.items()}
    for idx, v in enumerate(waveelems):
        x, y = inv_indx[idx]
        wavefront[y, x] = v

    return wavefront