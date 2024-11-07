from math import factorial

import numpy as np
from scipy.special import hermite

from .transverse_profile import TransverseProfile


class HermiteGaussianTransverseProfile(TransverseProfile):
    r"""
    A high-order Gaussian laser pulse expressed in the Hermite-Gaussian formalism.
    Definition is according to Siegman "Lasers" pg 646 eq 60 with the beam center
    upon the optical axis, :math:`x_0,y_0 = (0,0)`.

    More precisely, the transverse envelope (to be used in the
    :class:`.CombinedLongitudinalTransverseLaser` class) corresponds to:

    .. math::
        \mathcal{T}(x, y) = \,
                \mathcal{H}_m (x) \, \mathcal{H}_n(y) \, \exp(i \Phi)

    with

    .. math::
        \mathcal{H}_m(x) = A_m h_m \left ( \frac{\sqrt{2}x}{w_x(z)} \right) \exp{\left( -\frac{x^2}{w_x^2(z)}\right)} \exp{ \left ( -i k_0 \frac{x^2}{2 R_x(z)} \right )} 

        \mathcal{H}_n(y) = A_n h_n \left ( \frac{\sqrt{2}y}{w_y(z)} \right) \exp{\left( -\frac{y^2}{w_y^2(z)}\right)} \exp{ \left ( -i k_0 \frac{y^2}{2 R_y(z)} \right )} 

        w_x(z) = w_{0,x} \sqrt{1 + \left( \frac{z}{Z_x}\right)^2}

        w_y(z) = w_{0,y} \sqrt{1 + \left( \frac{z}{Z_y}\right)^2}

        A_m = \frac{1}{\sqrt{w_x(z) 2^{(m-1/2)} m!\sqrt{\pi}}}

        A_n = \frac{1}{\sqrt{w_y(z) 2^{(n-1/2)} n!\sqrt{\pi}}}

        R_x(z) = z + \frac{Z_x^2}{z}

        R_y(z) = z + \frac{Z_y^2}{z}

        \Phi(z) = \Phi_x(z) + \Phi_y(z)

        \Phi_x(z) = \left(m+\frac{1}{2}\right) \tan^{-1}\left({\frac{z}{Z_x}}\right)

        \Phi_y(z) = \left(n+\frac{1}{2}\right) \tan^{-1}\left({\frac{z}{Z_y}}\right)


    where  :math:`h_{n}` is the Hermite polynomial of order :math:`n`.

    Parameters
    ----------
    w_x : float (in meter)
        The waist of the laser pulse in the x direction, 
        i.e. :math:`w_x` in the above formula.
    w_y : float (in meter)
        The waist of the laser pulse in the y direction, 
        i.e. :math:`w_y` in the above formula.
    n_x : int (dimensionless)
        The order of hermite polynomial in the x direction
    n_y : int (dimensionless)
        The order of hermite polynomial in the y direction
    wavelength : float (in meter), optional
        The main laser wavelength :math:`\lambda_0` of the laser.
        (Only needed if ``z_foc`` is different than 0.)
    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at
        ``z=0``.)

    Warnings
    --------
    In order to initialize the pulse out of focus, you can either:

    - Use a non-zero ``z_foc``
    - Use ``z_foc=0`` (i.e. initialize the pulse at focus) and then call
      ``laser.propagate(-z_foc)``

    Both methods are in principle equivalent, but note that the first
    method uses the paraxial approximation, while the second method does
    not make this approximation.
    """

    def __init__(self, w_x, w_y, n_x, n_y, wavelength=None, z_foc=0):
        super().__init__()
        self.w_x = w_x
        self.w_y = w_y
        self.n_x = n_x
        self.n_y = n_y
        self.wavelength = wavelength
        self.z_foc = z_foc
        z_eval = - z_foc # this links our observation pos. to Siegmann's definition

        self.k0 = 2*np.pi/wavelength
    
        # Calculate Rayleigh Lengths
        Zx  = np.pi*w_x**2/wavelength
        Zy  = np.pi*w_y**2/wavelength
        
        # Calculate Size at Location Z
        wxZ = w_x*np.sqrt(1 + (z_eval/Zx)**2)
        wyZ = w_y*np.sqrt(1 + (z_eval/Zy)**2)
        
        # Calculate Multiplicative Factors
        Anx = 1 / np.sqrt( wxZ * 2**(n_x-1/2) * factorial(n_x) * np.sqrt(np.pi) )
        Any = 1 / np.sqrt( wyZ * 2**(n_y-1/2) * factorial(n_y) * np.sqrt(np.pi) )
        
        # Calculate the Phase contributions from propagation
        phiXz = (n_x + 1/2)*np.arctan2(z_eval,Zx)
        phiYz = (n_y + 1/2)*np.arctan2(z_eval,Zy)
        phiZ = phiXz + phiYz

        self.z_eval = z_eval
        self.Zx = Zx
        self.Zy = Zy
        self.wxZ = wxZ
        self.wyZ = wyZ
        self.Anx = Anx
        self.Any = Any
        self.phiZ = phiZ
        

    def _evaluate(self, x, y):
        """
        Return the transverse envelope.

        Parameters
        ----------
        x, y : ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """
        z_eval = self.z_eval
        Zx = self.Zx
        Zy = self.Zy
        k0 = self.k0
        wxZ = self.wxZ
        wyZ = self.wyZ
        Anx = self.Anx
        Any = self.Any
        n_x = self.n_x
        n_y = self.n_y
        phiZ = self.phiZ
        
        # Calculate the HG in each plane
        HGnx = Anx * hermite(n_x)( np.sqrt(2) * (x)/wxZ ) * np.exp(-(x)**2/wxZ**2) * np.exp( -1j * k0 * (x)**2/2/(z_eval**2 + Zx**2)*z_eval)
        HGny = Any * hermite(n_y)( np.sqrt(2) * (y)/wyZ ) * np.exp(-(y)**2/wyZ**2) * np.exp( -1j * k0 * (y)**2/2/(z_eval**2 + Zy**2)*z_eval)
    
        
        # Put it altogether
        envelope = HGnx * HGny * np.exp(1j*phiZ)
        

        return envelope
