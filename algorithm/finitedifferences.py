import numpy as np

def xy_to_rt(x, y, shape):
    rho = np.sqrt((x - shape[1]/2)**2 + (y - shape[0]/2)**2)
    theta = np.arctan2(y, x)
    return rho, theta

def solve_wavefront(laplacian, normal):
    waveheight = laplacian.shape[0]
    wavewidth = laplacian.shape[1]
    wavesize = wavewidth * waveheight

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
                print(r, f)

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
                #The sins and cosines only used if < 0

                rho, theta = xy_to_rt(x, y, laplacian.shape)

                r = [0] * numelems
                r[indicies_dict[(x,y)]] = 1
                if (x-1, y) in indicies_dict:
                    r[indicies_dict[(x-1, y)]] = min(0, -np.cos(theta))
                if (x+1, y) in indicies_dict:
                    r[indicies_dict[(x+1, y)]] = min(0,  np.cos(theta))
                if (x, y-1) in indicies_dict:
                    r[indicies_dict[(x, y-1)]] = min(0, -np.sin(theta))
                if (x, y+1) in indicies_dict:
                    r[indicies_dict[(x, y+1)]] = min(0,  np.sin(theta))
                M.append(r)
                f.append(normal[y,x])

    #Solve Matrix equation
    print(M)
    waveelems = np.linalg.solve(np.array(M), np.array(f))

    #Insert into wavefront
    wavefront = np.zeros_like(laplacian)
    inv_indx = {v: k for k, v in indicies_dict.items()}
    for idx, v in enumerate(waveelems):
        x, y = inv_indx[idx]
        wavefront[y, x] = v

    return wavefront