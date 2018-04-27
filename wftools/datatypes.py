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

    zernikefunctions = {
        1 : lambda rho, theta: 1,
        2 : lambda rho, theta: 2*rho*np.cos(theta),
        3 : lambda rho, theta: 2*rho*np.sin(theta),
        4 : lambda rho, theta: (3**0.5)*(2*rho*rho-1),
        5 : lambda rho, theta: (6**0.5)*rho*rho*np.sin(2*theta),
        6 : lambda rho, theta: (6**0.5)*rho*rho*np.cos(2*theta),
        7 : lambda rho, theta: (8**0.5)*(3*rho*rho*rho - 2*rho)*np.sin(theta),
        8 : lambda rho, theta: (8**0.5)*(3*rho*rho*rho - 2*rho)*np.cos(theta),
        9 : lambda rho, theta: (8**0.5)*rho*rho*rho*np.sin(3*theta),
        10: lambda rho, theta: (8**0.5)*rho*rho*rho*np.cos(3*theta),
        11: lambda rho, theta: (5**0.5)*(6*rho**4 - 6*rho*rho + 1),
        12: lambda rho, theta: (10**0.5)*(4*rho**4 - 3*rho*rho)*np.cos(2*theta),
        13: lambda rho, theta: (10**0.5)*(4*rho**4 - 3*rho*rho)*np.sin(2*theta),
        14: lambda rho, theta: (10**0.5)*(rho**4)*np.cos(4*theta),
        15: lambda rho, theta: (10**0.5)*(rho**4)*np.sin(4*theta)
    }

    def __init__(self, coef):
        self.coef = coef

    def at_wavelength(self, wavelength):
        return [z*wavelength for z in self.coef]

    def as_image(self, x_res, y_res, radius, zeroRim = False):
        # Generate X Y coords
        X = np.arange(-1, 1, 2 / x_res)
        Y = np.arange(-1 * y_res / x_res, 1 * y_res / x_res, 2 / y_res)
        X, Y = np.meshgrid(X, Y)

        #Generate theta rho coords
        theta = np.arctan2(Y, X)
        rho = np.sqrt(X ** 2 + Y ** 2)

        #Build the wavefront using the provided coefficients
        Z = np.zeros(X.shape)
        for idx, coeff in enumerate(self.coef):
            Z[rho<radius] += coeff * np.vectorize(self.zernikefunctions[idx + 1])(rho, theta)[rho<radius]

        if zeroRim:
            Z[rho > radius] = np.min(Z)
        return Z

    #Build a wavefront object from an amplitude map
    @classmethod
    def from_image(cls, image):
        y_res = image.shape[0]
        x_res = image.shape[1]

        nhot = np.sum(image > 0)

        # Generate X Y coords
        X = np.arange(-1, 1, 2 / x_res)
        Y = np.arange(-1, 1, 2 / y_res)
        X, Y = np.meshgrid(X, Y)

        coefficients = [0 for i in range(15)]

        pix_count = 0

        for xr, yr, zr in zip(X, Y, image):
            for x, y, z in zip(xr, yr, zr):
                rho = np.sqrt(x ** 2 + y ** 2)
                theta = np.arctan2(y, x)
                if z == 0:
                    continue
                coefficients = [coefficients[i] + z * cls.zernikefunctions[i + 1](rho, theta) for i in range(6)]
                pix_count += 1

        return cls([coef / pix_count**2.5 for coef in coefficients])

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

