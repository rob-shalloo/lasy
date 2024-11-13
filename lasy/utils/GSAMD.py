import numpy as np
from scipy.special import hermite
import laserbeamsize as lbs
import time
from scipy.optimize import minimize
from scipy.constants import c, epsilon_0
try:
    from optimas.core import VaryingParameter, Objective
    from optimas.generators import GridSamplingGenerator 
    from optimas.generators import AxSingleFidelityGenerator
    from optimas.evaluators import FunctionEvaluator
    from optimas.explorations import Exploration
except:
    print('No Optimas installed')

class GerchbegSaxtonWithModalDecomposition():
    """
    An implementation of the Gerchberg Saxton Algorithm with Modal Decomposition (GSA-MD)
    as described by I. Moulanier et al., Jour. Opt. Soc. Am. B 40, 9 (2023). DOI 10.1364/JOSAB.489884
    
    """
    
    def __init__(self,x,y,z,z0,inten,nxMax,nyMax,wavelength,NGS,NCenter,showProgress=False):
        """
        Parameters
        ----------
        x, y: ndarrays of floats (meters)
            Points on the transverse plane upon which to intensity is evaluated. These 
            arrays need to all have the same shape although can be either 1d arrays or a 
            set of 2D arrays created with numpy.meshgrid

        z: ndarray of floats (meters)
            Longitudinal (along the propagation axis) positions at which to the intensity
            distributions have been evaluated. The array should be 1D. The array should be ordered
            such that it is monotonically increasing. 

        z0: float (meters)
            The estimated location of the focal plane of the laser

        inten: ndarray of floats (meters)
            3D array of floats containing several (minimum 2) 2D measurements of the laser intensity.
            axis 0 is the y-axis, axis 1 is the x-axis and axis 2 is the z-axis. The length of each
            axis should match the x,y and z arrays above. The ordering of axis 2 should be such that the
            intensity profiles at each index in axis 2 match the z location idenitified at the same index
            in the z-array. 

        nxMax,nyMax: ints 
            The maximum order of the mode along the x and y axes that will be used in the
            decomposition

        NGS, NCenter: ints
            The maximum number of itterations to take for the Gerchberg Saxton and centering portions of 
            the algorithm respectively

        wavelength: float (meters)
            The wavelength of the laser

        showProgress: bool
            If true the the function will print regular updates on how the algorithm is progressing.

        """
        
        # find the index of the focus and Roll the z array and the inten array to place 
        # the focal plane at the start
        z0Indx = np.argmin(np.abs(z-z0))
        z = np.roll(z,-z0Indx)
        inten = np.roll(inten,-z0Indx,axis=2)
        _,_,Nimgs = inten.shape

        # Ensure focus is at z=0
        z = z-z0
        
        self.x = x
        self.y = y
        self.z = z
        self.z0 = z0
        self.inten = inten
        self.nxMax = nxMax
        self.nyMax = nyMax
        self.wavelength = wavelength
        self.NGS = NGS
        self.NCenter = NCenter
        self.showProgress = showProgress
        
        
        self.ittGSEval = 5
        self.chi2GradThreh = 0.02
        
        self.chi2 = 9e99
        
        self.approxPulseDuration = 30e-15
        
        
    def initialiseCenters(self,centers=None):
        """
        Initialise the centers for each longitudinal location
        
        Parameters
        ----------
        centers: ndarray of floats (meters)
            The location of the center of the intensity profile (for HG reconstruction)
            at each longitudinal location. This array will have a shape of (len(self.z),2)
            where the elements of centers[:,0] correspond to the y-location of the center and
            centers[:,1] correspond t the x-location of the centers. If none is provided then
            a uniform set of centers will be initialised based on a center of mass fit to the
            focus.
            
        """
        
        if centers is None:
            x0init, y0init, _, _, _ = lbs.beam_size(self.inten[:,:,0])
            x0init  = x0init * np.abs(np.mean(np.diff(self.x))) +np.min(self.x)        
            y0init  = y0init * np.abs(np.mean(np.diff(self.y))) +np.min(self.y)
            
            if self.showProgress:
                print("Estimated Center (x0init,y0init) : (%.1e, %.1e) m" %(x0init,x0init))
        
            centers = np.ones((len(self.z),2))
            centers[:,0] *= y0init
            centers[:,1] *= x0init
        
        self.centers = centers
        
        
    def initialiseSpotSizes(self,spotSizes=None):
        """
        Initialise the spot size of the modes at focus in each transverse direction
        
        Parameters
        ----------
        spotSizes: tuple of floats (meters)
            The size of the mode along the x and y axes at focus. spotSize[0] corresponds to 
            the y-axis while spotSize[1] corresponds to the x-axis.  If none is provided then
            the spot sizes will be initialised based on a D4sigma fit to the
            focus.
            
        """
        
        if spotSizes is None:
            _, _, dx, dy, _ = lbs.beam_size(self.inten[:,:,0])
            wx      = dx * np.abs(np.mean(np.diff(self.x)))/2
            wy      = dy * np.abs(np.mean(np.diff(self.y)))/2
            if self.showProgress:
                print("Estimated SpotSize (wx,wy) : (%.1e, %.1e) m" %(wx,wy))
                
            spotSizes = (wy,wx)
            
        self.spotSizes = spotSizes
        
        
    def cleanRawData(self,nxMaxClean=50,nyMaxClean=50):
        """
        This will perform a nxMax x myMax = (50 x 50) hermite gauss reconstruction of the
        provided laser intensity profiles to clean the data and remove blemishes such
        as dust spots, while also reducing noise in the images
        
        Parameters
        ----------
        nxMaxClean,nyMaxClean: ints
            The maximum order of the mode along the x and y axes that will be used in the
            decomposition and reconstruction for cleaning the data
        """
        
        inten = self.inten
        self.inten_raw = inten
        x = self.x
        y = self.y
        z = self.z
        centers = self.centers
        spotSizes = self.spotSizes
        wavelength = self.wavelength
        
        
        for k in range(len(z)):
            if self.showProgress: print('Cleaning Input Intensity Profile %i of %i' %(k+1,len(z)))
            # Decompose the intensity profile
            Cxy = {}
            for i in range(nxMaxClean):
                for j in range(nyMaxClean):
                    Cxy[(i,j)] = getHGMode_IntensityOnly(x,y,0,np.sqrt(inten[:,:,k]),centers[k,1],centers[k,0],spotSizes[1],spotSizes[0],i,j,wavelength)
            
            # Now reconstruct
            amplitudeNew = np.zeros((len(y),len(x)))
            for nxy in list(Cxy):
                amplitudeNew += Cxy[nxy]*np.real(HGAnalytic(x,y,0,centers[k,1],centers[k,0],spotSizes[1],spotSizes[0]
                                                           ,nxy[0],nxy[1],wavelength))
            self.inten[:,:,k] = np.abs(amplitudeNew)**2
        
    def estimateInitialCxy(self):
        """
        Provides a first estimte for the complex amplitudes of the modal decomposition
        
        """
        
        # Esimate Phase
        focalPlaneUncertainty = self.z[1]-self.z[0]
        phi = estimatePhi0(self.x,self.y,self.centers[0,1],self.centers[0,0],
                           np.mean(self.spotSizes),self.wavelength,focalPlaneUncertainty)

        # Construct Initial Guess of Field
        E0 = np.sqrt(2/c/epsilon_0*self.inten[:,:,0])*np.exp(1j*phi)

        # Find estimate of the decomposition of the initial electric field
        self.Cxy = decomposeHG(self.x,self.y,self.z[0],E0,self.centers[0,1],self.centers[0,0],
                               self.spotSizes[1],self.spotSizes[0],self.nxMax,self.nyMax,self.wavelength)
    
    def updateCxyEstimate(self,centerArr=None):
        """
        
        
        """
        Cxy = self.Cxy
        x = self.x
        y = self.y
        z = self.z
        
        if centerArr is None:
            centers = self.centers
        else:
            centers = np.zeros((int(len(centerArr)/2),2))
            for n in range(int(len(centerArr)/2)):
                centers[n,0] = centerArr[2*n]
                centers[n,1] = centerArr[2*n+1]
                
        spotSizes = self.spotSizes
        wavelength = self.wavelength
        nxMax = self.nxMax
        nyMax = self.nyMax
        Nimgs = len(z)
        NGS = self.NGS
        ittGSEval = self.ittGSEval
        chi2GradThreh = self.chi2GradThreh
        showProgress = self.showProgress
        
        Ftot = 0
        for key in Cxy:
            Ftot += np.abs(Cxy[key])**2
        
        for i in range(NGS):
            chi2 = 0
            if showProgress: print("GSA Iterration: %i" %i)
            for k in np.linspace(Nimgs-1,0,Nimgs).astype(int):
                # This loops backwards through the images and ensures that the last image
                # to be corrected is always the focus
                
                if showProgress: print("    Image: %i of %i" %(k+1,Nimgs))

                # Step 2
                t0 = time.time()
                Efield = reconstructHG(x,y,z[k],centers[k,1],centers[k,0],spotSizes[1],spotSizes[0],Cxy,wavelength)
                t1 = time.time()
                if showProgress: print("        Reconstruction: %.3f s" %(t1-t0))
                
                # Step 3
                phi = np.angle(Efield)

                # Step 4
                Enew = np.sqrt(2/c/epsilon_0*self.inten[:,:,k]) * np.exp(1j*phi)

                # Step 5
                delta = (np.sqrt(2/c/epsilon_0*self.inten[:,:,k]) - np.abs(Efield))/np.max(np.sqrt(2/c/epsilon_0*self.inten[:,:,k]))
                Enew *= np.exp(delta)

                # Step 6
                t0 = time.time()
                CxyNew = decomposeHG(x,y,z[k],Enew,centers[k,1],centers[k,0],spotSizes[1],spotSizes[0],
                                     nxMax,nyMax,wavelength)
                t1 = time.time()
                if showProgress: print("        Decomposition:  %.3f s" %(t1-t0))
                
                
                # Step 7
                coeffSum = 0
                for key in CxyNew:
                    coeffSum += np.abs(CxyNew[key])**2
                for key in CxyNew:
                    CxyNew[key] *= np.sqrt(Ftot/coeffSum)#.astype(complex)

                # Step 8
                for key in Cxy:
                    Cxy[key] = 1/2 * ( Cxy[key] + CxyNew[key])

                # Step 9
                coeffSum = 0 

                for key in Cxy:
                    coeffSum += np.abs(Cxy[key])**2
                for key in Cxy:
                    Cxy[key] *= np.sqrt(Ftot/coeffSum)#.astype(complex)
                
                Cxy = CxyNew
                # calculate the fluence error
                chi2 += np.sqrt(np.sum((np.abs(Enew)**2  - 2/c/epsilon_0*self.inten[:,:,k]) **2))/np.sum(2/c/epsilon_0*self.inten[:,:,k])/Nimgs


            if i%ittGSEval==0:
                if i > ittGSEval:
                    chi2Grad = (chi2 -chi2Old)/chi2Old
                    if showProgress:
                        print("chi2     = %.2e " %(chi2))
                        print("chi2Grad = %.2f %%" %(chi2Grad*100))
                    if np.abs(chi2Grad) < chi2GradThreh:
                        break
                chi2Old = chi2
        
        self.chi2 = chi2
        self.Cxy = Cxy
        
        return chi2
        
        
        
    def initialiseCenterSearch(self,dx0=None,dy0=None):
        """
        Initialise the ranges over which the center finding algorithm will search
        
        Parameters
        ----------
        dx0,dy0: float (meters)
            The full range over which the spot center can vary in x and y
        """
        if dx0 is None:
            dx0 = self.spotSizes[1]
        if dy0 is None:
            dy0 = self.spotSizes[0]    
            
        self.dx0 = dx0
        self.dy0 = dy0
        
        
    def GSAMD(self,bounds=None):
        """
        Perform the full Gerchberg Saxton with Modal Decomposition algorithm including
        both the itterative modal decompositions together with center finding algorithm.
        The center finding is attempted through Bayesian Optimisation powered by Optimas.
        
        Parameters
        ----------
        bounds: numpy array of two floats (in meters)
            provides the bounding range over which the spots will be moved during the centering
            algorithm. If not provided, then these default to the +/- the spot size at focus. 
        
        """
        
        _,_,NImgs = gs.inten.shape

        
        # Get the initial Estimate of the centers
        centers = self.centers
        centerArr0 = np.zeros(2*NImgs)
        bounds = []
        
        for i in range(NImgs):
            # center ordering y0,x0,y1,x1,y2,x2,...
            centerArr0[2*i] = centers[i,0] 
            centerArr0[2*i+1] = centers[i,1]
            bounds.append((-self.dy0/2,self.dy0/2))
            bounds.append((-self.dx0/2,self.dx0/2))
            
        # Provide the optimsation bounds
        if bounds is None:
            bounds = np.asarray([-np.mean(gs.spotSizes),np.mean(gs.spotSizes)])
          

        # Unit Conversions for Optimisation Algorithm
        # We use microns here
        bounds = bounds*1e6
        
        
        # Create the optimisation variables
        variables = []
        for i in range(NImgs):
            variables.append(VaryingParameter("x%i"%i, bounds[0], bounds[1]))
            variables.append(VaryingParameter("y%i"%i, bounds[0], bounds[1]))

        obj = Objective("f")
        
        # Create generator.
        gen = AxSingleFidelityGenerator(
            varying_parameters=variables, objectives=[obj], n_init=4
        )

        # Create evaluator.
        ev = FunctionEvaluator(function=runGSA)

        # Create exploration.
        exp = Exploration(
            generator=gen, evaluator=ev, max_evals=NCenter, sim_workers=4, run_async=True
        )

        # Run the optimisation
        exp.run()
        
        # Get the optimum parameter set
        # ???
        
        return None
        
        
        
    def reconstructField(self,x,y,z,x0=0,y0=0):
        """
        Reconstruct the electric field distribution of the laser


        Parameters
        ----------
        x, y: ndarrays of floats (meters)
            Define points on the transverse plane upon which to evaluate the field. These 
            arrays need to all have the same shape although can be either 1d arrays or a 
            set of 2D arrays created with numpy.meshgrid

        z: float (meters)
            Longitudinal (along the propagation axis) position at which to evaluate the 
            field. 

        Returns
        -------
        field: ndarray of complex floats 
            The reconstructed electric field 

        """
        spotSizes = self.spotSizes
        Cxy = self.Cxy
        wavelength = self.wavelength
        
        field = reconstructHG(x,y,z,x0,y0,spotSizes[1],spotSizes[0],Cxy,wavelength)
        
        return field
    
    
def _updateCxyEstimate(centerArr,obj):
    """
    a wrapper for the center optimisation
    """
    chi2 = obj.updateCxyEstimate(centerArr)
    
    return chi2
    

    
def HGAnalytic(x,y,z,x0,y0,wx,wy,nx,ny,wavelength):
    """
    Evaluates a complex Hermite-Gauss according to eqn3 of DOI: 10.1364/JOSAB.489884
    Specificaly this representation allows for the computation of the mode with distinct
    mode widths in x and y while also allows the mode to be calculated at an arbitrary
    location along the propagation axis z
    
    Parameters
    ----------
    x, y: ndarrays of floats (meters)
        Define points on the transverse plane upon which to evaluate the field. These 
        arrays need to all have the same shape, 1d.
    
    z: float (meters)
        Longitudinal (along the propagation axis) position at which to evaluate the 
        field. 
        
    x0,y0: floats (meters)
        The center position of the mode in the x and y plane
        
    wx,wy: floats (meters)
        The size of the mode along the x and y axes at focus
        
    nx,ny: ints 
        The order of the mode along the x and y axes
        
    wavelength: float (meters)
        The wavelength of the laser
        
    Returns
    -------
    HG: ndarray of complex floats
        The field of the evaluated Hermite Gauss mode
        
    """
    
    assert len(x.shape)==1
    assert len(y.shape)==1
    
    k0 = 2*np.pi/wavelength
    
    # Calculate Rayleigh Lengths
    Zx  = np.pi*wx**2/wavelength
    Zy  = np.pi*wy**2/wavelength
    
    # Calculate Size at Location Z
    wxZ = wx*np.sqrt(1 + (z/Zx)**2)
    wyZ = wy*np.sqrt(1 + (z/Zy)**2)
    
    # Calculate Multiplicative Factors
    Anx = 1 / np.sqrt( wxZ * 2**(nx-1/2) * np.math.factorial(nx) * np.sqrt(np.pi) )
    Any = 1 / np.sqrt( wyZ * 2**(ny-1/2) * np.math.factorial(ny) * np.sqrt(np.pi) )
    
    # Calculate the Phase contributions from propagation
    phiXz = (nx + 1/2)*np.arctan2(z,Zx)
    phiYz = (ny + 1/2)*np.arctan2(z,Zy)
    phiZ = phiXz + phiYz
    
    # Calculate the HG in each plane
    hgnx = Anx * hermite(nx)( np.sqrt(2) * (x-x0)/wxZ ) * np.exp(-(x-x0)**2/wxZ**2) * np.exp( -1j * k0 * (x-x0)**2/2/(z**2 + Zx**2)*z)
    hgny = Any * hermite(ny)( np.sqrt(2) * (y-y0)/wyZ ) * np.exp(-(y-y0)**2/wyZ**2) * np.exp( -1j * k0 * (y-y0)**2/2/(z**2 + Zy**2)*z)
    
    HGnx,HGny = np.meshgrid(hgnx,hgny)
    # Put it altogether
    HG = HGnx * HGny * np.exp(1j*phiZ)
    
    return HG

def estimatePhi0(x,y, x0,y0, w0,wavelength,deltaZ):
    """
    Estimate the phase of a Gaussian at a short distance away from the focal plane.
    This is used as an initial estimate of the phase of the pulse and helps with convergence
    of the algorithm.
    
    Parameters
    ----------
    x, y: ndarrays of floats (meters)
        Define points on the transverse plane upon which to evaluate the phase. These 
        arrays need to all have the same shape although can be either 1d arrays or a 
        set of 2D arrays created with numpy.meshgrid
    
    x0,y0: floats (meters)
        The center position of the mode in the x and y plane
        
    wavelength: float (meters)
        The wavelength of the laser
        
    deltaZ: float (meters)
        An estimate for the uncertainty in the focal plane of the laser
    
    """
        
    k0 = 2*np.pi/wavelength
    
    phi0 = k0 * ((x-x0)**2 + (y-y0)**2)/(2*deltaZ*( 1 + (k0*w0**2/2/deltaZ)**2))
    
    return phi0
    
    
def getHGMode(x,y,z,field,x0,y0,wx,wy,nx,ny,wavelength):
    """
    Determine, via projection, the complex amplitude of a single Hermite Gauss mode in 
    a provided electric field
    
    Parameters
    ----------
    x, y: ndarrays of floats (meters)
        Points on the transverse plane upon which to field is evaluated. These 
        arrays need to all have the same shape although can be either 1d arrays or a 
        set of 2D arrays created with numpy.meshgrid
    
    z: float (meters)
        Longitudinal (along the propagation axis) position at which the field is evaluated.
        
    field: ndarray of complex floats 
        The electric field onto which the Hermite Gauss mode should be projected
        
    x0,y0: floats (meters)
        The center position of the mode in the x and y plane
        
    wx,wy: floats (meters)
        The size of the mode along the x and y axes at focus
        
    nx,ny: ints 
        The order of the mode along the x and y axes
        
    wavelength: float (meters)
        The wavelength of the laser
        
    Returns
    -------
    coeff: complex float
        The complex amplitude of the selected Hermite Gauss mode within the provided 
        electric field
    
    """
    
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    
    hg = HGAnalytic(x,y,z,x0,y0,wx,wy,nx,ny,wavelength)
    
    coeff = np.sum(field*np.conj(hg))*dx*dy
    
    return coeff

def getHGMode_IntensityOnly(x,y,z,amplitude,x0,y0,wx,wy,nx,ny,wavelength):
    """
    Determine, via projection, the amplitude of a single Hermite Gauss mode in 
    a provided amplitude profile field. Should be used for data cleaning purposes
    rather than for the full implementation of GSA-MD
    
    Parameters
    ----------
    x, y: ndarrays of floats (meters)
        Points on the transverse plane upon which to field is evaluated. These 
        arrays need to all have the same shape although can be either 1d arrays or a 
        set of 2D arrays created with numpy.meshgrid
    
    z: float (meters)
        Longitudinal (along the propagation axis) position at which the field is evaluated.
        
    amplitude: ndarray of floats 
        The square root of the intensity if the field onto which the Hermite Gauss mode 
        should be projected
        
    x0,y0: floats (meters)
        The center position of the mode in the x and y plane
        
    wx,wy: floats (meters)
        The size of the mode along the x and y axes at focus
        
    nx,ny: ints 
        The order of the mode along the x and y axes
        
    wavelength: float (meters)
        The wavelength of the laser
        
    Returns
    -------
    coeff: complex float
        The complex amplitude of the selected Hermite Gauss mode within the provided 
        electric field
    
    """
    
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    
    hg = HGAnalytic(x,y,z,x0,y0,wx,wy,nx,ny,wavelength)
    
    coeff = np.real(np.sum(amplitude*hg)*dx*dy)
    
    return coeff
    
def decomposeHG(x,y,z,field,x0,y0,wx,wy,nxMax,nyMax,wavelength):
    """
    Perform a modal decomposition of the provided electric field distribution into 
    a set of Hermite Gauss modes
    
    Parameters
    ----------
    x, y: ndarrays of floats (meters)
        Points on the transverse plane upon which to field is evaluated. These 
        arrays need to all have the same shape although can be either 1d arrays or a 
        set of 2D arrays created with numpy.meshgrid
    
    z: float (meters)
        Longitudinal (along the propagation axis) position at which the field is evaluated.
        
    field: ndarray of complex floats 
        The electric field onto which the Hermite Gauss mode should be projected
        
    x0,y0: floats (meters)
        The center position of the mode in the x and y plane
        
    wx,wy: floats (meters)
        The size of the mode along the x and y axes at focus
        
    nxMax,nyMax: ints 
        The maximum order of the mode along the x and y axes that will be used in the
        decomposition
        
    wavelength: float (meters)
        The wavelength of the laser
        
    Returns
    -------
    cxy: structure of complex float
        The complex amplitude of the each of the Hermite Gauss modes within the
        decomposition. Each mode amplitude is indexed by a tuple (nx,ny) in which
        nx and ny represent the order of the mode along each axis.
    
    """
    cxy = {}
    for i in range(nxMax):
        for j in range(nyMax):
            cxy[(i,j)] = getHGMode(x,y,z,field,x0,y0,wx,wy,i,j,wavelength)
        
    return cxy

def reconstructHG(x,y,z,x0,y0,wx,wy,cxy,wavelength):
    """
    Reconstruct an electric field distribution from a set of complex HG mode amplitudes
    
    
    Parameters
    ----------
    x, y: ndarrays of floats (meters)
        Define points on the transverse plane upon which to evaluate the field. These 
        arrays need to all have the same shape although can be either 1d arrays or a 
        set of 2D arrays created with numpy.meshgrid
    
    z: float (meters)
        Longitudinal (along the propagation axis) position at which to evaluate the 
        field. 
        
    x0,y0: floats (meters)
        The center position of the mode in the x and y plane
        
    wx,wy: floats (meters)
        The size of the mode along the x and y axes at focus
        
    cxy: structure of complex float
        The complex amplitude of the each of the Hermite Gauss modes within the
        decomposition. Each mode amplitude is indexed by a tuple (nx,ny) in which
        nx and ny represent the order of the mode along each axis.
        
    wavelength: float (meters)
        The wavelength of the laser
        
    Returns
    -------
    field: ndarray of complex floats 
        The reconstructed electric field 
    
    """

    field = 1j*np.zeros((len(y),len(x)))
    
    for nxy in list(cxy):
        field += cxy[nxy]*HGAnalytic(x,y,z,x0,y0,wx,wy,nxy[0],nxy[1],wavelength)
        
    return field


    
    
    
    
    
    
    
    