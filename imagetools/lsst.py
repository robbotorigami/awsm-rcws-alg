
def formatpixel(f):
    s = "{:.3E}".format(f/128)
    mantissa, exp = s.split('E')
    # add 1 to digits as 1 is taken by sign +/-
    return  "{:.3f}E{:+04}".format(abs(float(mantissa)), int(exp))

def save_to_file(pre_im, pos_im, basename):
    for row in pre_im:
        for pix in row:
            if pix < 0:
                raise Exception("Fatal error, pixel < 0")
    for row in pos_im:
        for pix in row:
            if pix < 0:
                raise Exception("Fatal error, pixel < 0")
    prestr = '\n'.join([' '.join([formatpixel(p) for p in line[:-1]]) for line in pre_im[:-1]])
    poststr = '\n'.join([' '.join([formatpixel(p) for p in line[:-1]]) for line in pos_im[:-1]])
    with open(basename + "_pre" + ".txt", 'w') as f:
        f.write(prestr)
    with open(basename + "_post" + ".txt", 'w') as f:
        f.write(poststr)