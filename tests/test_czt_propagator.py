"""Test the implementation of the CZT Propagator

Performs a resampling propagation of a gaussian beam after focusing by a lens for two seperate new grid sizes and compares the resulting beam size
with that expected analytically
"""
import numpy as np
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.laser import Laser
from lasy.optical_elements.parabolic_mirror import ParabolicMirror
from lasy.utils.grid import Grid
import matplotlib.pyplot as plt

def test_czt_propagator():
    # Parameters
    w0Initial = 20e-3
    wavelength = 8e-7
    xRange = (-100e-3,100e-3)
    xFRange = (-100e-6,100e-6)

    pol=(1,0)
    laser_energy = 1.0
    tau = 30e-15
    t_peak=0.0

    dim = 'xyt'
    lo = (xRange[0],xRange[0],-100e-15)
    loF = (xFRange[0],xFRange[0],lo[2])

    hi = (xRange[1],xRange[1],100e-15)
    hiF = (xFRange[1],xFRange[1],hi[2])
    npoints = (512,512,100)

    focal_length = 1.0

    newGrid = Grid(dim,lo=loF,hi=hiF,npoints=npoints)


    profile = GaussianProfile(wavelength, pol, laser_energy, w0Initial, tau, t_peak)
    laser = Laser(dim, lo, hi, npoints, profile)

    OAP = ParabolicMirror(focal_length)

    laser.apply_optics(OAP)

    laser.propagate(focal_length,grid=newGrid)

    laser.show()
    plt.ion()
    plt.show()

    