import poppy
import matplotlib.pyplot as plt

RADIUS = 1.0 # meters
WAVELENGTH = 460e-9 # meters
PIXSCALE = 0.01 # arcsec / pix
FOV = 1 # arcsec
NWAVES = 1.0

plt.figure(figsize=(18,2))

results = []

for coefficient_set in [ [0, 0, 0, 0, 35e-9]]:
    osys = poppy.OpticalSystem()
    circular_aperture = poppy.CircularAperture(radius=RADIUS)
    osys.add_pupil(circular_aperture)
    zwfe = poppy.ZernikeWFE(
        coefficients=coefficient_set,
        radius=RADIUS
    )
    osys.add_pupil(zwfe)
    osys.add_detector(pixelscale=PIXSCALE, fov_arcsec=FOV)

    psf = osys.calc_psf(wavelength=WAVELENGTH, display=False)
    results.append(psf)
plt.show()
