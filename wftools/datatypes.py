import numpy as np

class zernike_wfe:
    names = ['1_PISTON',
             '2_X tilt',
             '3_Y tilt',
             '4_Focus',
             '5_45DegAstigmatism',
             '6_0DegAstigmatism',
             '7_YComa',
             '8_XComa',
             '9_YTrefoil',
             '10_XTrefoil',
             '11_Spherical']

    def __init__(self, coef):
        self.coef = coef

    def at_wavelength(self, wavelength):
        return [z*wavelength for z in self.coef]

class optical_setup:

    def __init__(self, pixel_size, focal_length, wavelength, defocus, aperature):
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.defocus = defocus
        self.aperature = aperature

    def sensor_height(self):
        return self.defocus*self.aperature/self.focal_length

    def wavelength_as_um(self):
        return self.wavelength*1e6

