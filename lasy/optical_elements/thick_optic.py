import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement
import matplotlib.pyplot as plt

class ThickOptic(OpticalElement):
    r"""
    Class for a thick radially symmetric optic.

    The absolute location of the two surfaces is unimportant, rather the relative
    difference in their locations is utilised.
    
    The optic is assumed to be located in vacuum, with the surrounding medium having
    a refractiev index of 1.

    Parameters:
    -----------
    radius : ndarray of floats (in meter)
        Defines the radial points on which the optical surfaces are evaluated.

    surface1 : ndarray of floats or callable
        Either an array of length equal to `radius` or a callable function to be
        evaluated on `radius`. It defines the locatioon of the first surface with which
        the beam interacts in longitudinal coordinates.

    surface2 : ndarray of floats or callable
        Either an array of length equal to `radius` or a callable function to be
        evaluated on `radius`. It defines the locatioon of the second surface with which
        the beam interacts in longitudinal coordinates.

    eta_material : float or callable
        The real component of the refractive index of the material from which the optic
        is manufactured. If a float, a single value is used. If callable, then the callable
        should be a function of frequency `omega`.

    """

    def __init__(self, radius, surface1, surface2, eta_material):
        super().__init__()

        # if surface1 or surface2 are callables then create arrays
        if callable(surface1):
            s1 = surface1(radius)
        else:
            s1 = surface1
        if callable(surface2):
            s2 = surface2(radius)
        else:
            s2 = surface2

        # Get maximum distance between the surfaces
        max_distance = np.max(s2) - np.min(s1)

        # Calculate the path length of material and length of vacuum
        # through which the beam passes
        material_length = s2-s1
        vacuum_length = max_distance - material_length
        
        assert np.all(vacuum_length >=0)
        assert np.all(material_length >=0)

        self.radius = radius
        self.s1 = s1
        self.s2 = s2
        self.vacuum_length = vacuum_length
        self.material_length = material_length

        self.eta_material = eta_material
        self.eta_vacuum = 1

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
        radius = self.radius
        vacuum_length = self.vacuum_length
        material_length = self.material_length
        eta_material = self.eta_material
        eta_vacuum = self.eta_vacuum

        r = np.sqrt(x**2 + y**2)

        if callable(eta_material):
            eta_material_eval = eta_material(omega+omega0)
        else:
            eta_material_eval = eta_material

        # calculate the phase shifts
        phase_shift_material = (omega+omega0) / c * eta_material_eval * np.interp(r,radius,material_length) 
        phase_shift_vacuum   = (omega+omega0) / c *     eta_vacuum    * np.interp(r,radius,vacuum_length) 
        
        
        return np.exp(1j * (phase_shift_material + phase_shift_vacuum))
    
    def plot_optic(self,omega0,Npoints=1000):
        """
        Creates a plot of the surfaces of the optic for diagnostic purposes

        omega0 : float (rad/s)
            central frequency of beam.
        Npoints : int 
            Number of points over which to plot the results
        """
        radius =  np.linspace(self.radius[0],self.radius[-1],Npoints)
        s1 = np.interp(radius,self.radius,self.s1)
        s2 = np.interp(radius,self.radius,self.s2)

        amplitude_multiplier = self.amplitude_multiplier(radius,np.zeros_like(radius),0,omega0)
        phase_shift = np.angle(amplitude_multiplier)

        fig,ax =plt.subplots(1,2,tight_layout=True)

        ax[0].plot(1e3*np.concatenate((s1[::-1],s1)),1e3*np.concatenate((-radius[::-1],radius)),color=(0,0,0))
        ax[0].plot(1e3*np.concatenate((s2[::-1],s2)),1e3*np.concatenate((-radius[::-1],radius)),color=(0,0,0))
        ax[0].hlines(-1e3*radius[-1],1e3*s1[-1],1e3*s2[-1],color=(0,0,0))
        ax[0].hlines(1e3*radius[-1],1e3*s1[-1],1e3*s2[-1],color=(0,0,0))
        ax[0].set_ylim(-1.1*1e3*np.max(radius),1.1*1e3*np.max(radius))
        ax[0].set_xlim(1e3*np.min(s1)-1e3*np.max(s2/10),1.1*1e3*np.max(s2))

        ax[0].fill_betweenx(1e3*np.concatenate((-radius[::-1],radius)),1e3*np.concatenate((s1[::-1],s1)),1e3*np.concatenate((s2[::-1],s2)),color=(0.5,0.5,1,.1))

        ax[0].set_xlabel('Optical Axis (mm)')
        ax[0].set_ylabel('Transverse Axis (mm)')

        ax[1].set_title('Phase Shift Cross Section')
        ax[1].plot(np.concatenate((phase_shift[::-1],phase_shift)),1e3*np.concatenate((-radius[::-1],radius)))
        ax[1].set_xlabel('Phase Shift (rad)')
        ax[1].set_ylabel('Transverse Axis (mm)')
        ax[1].set_ylim(-1.1*1e3*np.max(radius),1.1*1e3*np.max(radius))

        print("Peak to Valley Phase Shift = %.2f pi radians" %((np.max(phase_shift)-np.min(phase_shift))/2/np.pi ))
