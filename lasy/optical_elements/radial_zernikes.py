import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement


class RadialZernikes(OpticalElement):
    """
    Class for adding Radial Zernikes for example from A Deformable Mirror.

    The Zernikes are indentified by their OSA/ANSI indexing.
    Thus:
        z4 = focus
        z12 = primary spherical
        z22 = secondary spherical


    Parameters
    ----------
    R : float (in meter)
        The radius of the pupil used for defining the Zernikes.

    z4 : float
        Amplitude of the focus term

    z12 : float
        Amplitude of the primary spherical term

    z22 : float
        Amplitude of the secondary spherical term

    """

    def __init__(self, R, z4, z12, z22):
        self.R = R
        self.z4 = z4
        self.z12 = z12
        self.z22 = z22

    def amplitude_multiplier(self, x, y, omega, omega0):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.
        omega0 : float (in rad/s)
            Central angular frequency, as used for the definition
            of the laser envelope.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        rho                 = np.sqrt(x**2 + y**2)/self.R
        focalTerm           = np.sqrt(3) * ( 2*rho**2 - 1)
        primarySpherical    = np.sqrt(5) * ( 6*rho**4 - 6*rho**2 + 1)
        secondarySpherical  = np.sqrt(7) * ( 20*rho**6 - 30*rho**4 + 12*rho**2 - 1) 

        phase = omega/ c / np.pi * (self.z4 * focus + self.z12 * primarySpherical + self.z22 * secondarySpherical)
        return np.exp(-1j * phase)
