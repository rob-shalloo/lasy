import numpy as  np
from scipy.signal import zoom_fft

def zoomFourierTransform2D(x,y,f,kxRange,kyRange):
    """ 
    Performs a zoomed fourier transform on the complex data f

    Parameters
    ----------
    x,y : numpy arrays (meters)
        Spatial arrays upon which the complex laser field `f` is defined.

    f : ndarray 
        the complex array to be fourier transformed
    
    kxRange,kyRange: tuples (rad/m)
        The desired frequency range within which to calculate the fourier transform

    Returns
    -------
    k_x, k_y: numpy arrays (rad/meter)
        The frequency upon which the fourier transform is calculated
    F: ndarray
        The fourier transform complex amplitude
    
    """
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    m,n = np.shape(f)
        
    F = zoom_fft(zoom_fft(f, kyRange, n, fs=2*np.pi/dy, axis=0), kxRange, m, fs=2*np.pi/dx, axis=1)#*dx*dy

    # Calculate Frequency Axes. The 2pi is because we're returning the axis like k_x rather than 1/x
    k_x = np.linspace(kxRange[0],kxRange[1],m)
    k_y = np.linspace(kyRange[0],kyRange[1],n)

    return (k_x,k_y,F)

def propagateCZT(x,y,field,wavelength,z,xFRange,yFRange):
    """
    Propagate the laser using the Chirped Z-Transform method
    [Hu, Y., Wang, Z., Wang, X. et al. 
    Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method. 
    Light Sci Appl 9, 119 (2020). https://doi.org/10.1038/s41377-020-00362-z]

    Parameters:
    -----------
    x,y: 1d numpy arrays (meters)
        Defining the spatial extent over which the field is calculated in the input plane

    field: 2d numpy array (complex E-Field)
        Representing the complex electric field profile to be diffracted. This is the field at the input plane
    
    wavelength: float (meters)
        The wavelength of light being diffracted

    z: float (meters)
        The distance over which the diffraction should be calculated

    xFRAnge: tuple (meters)
        The x-range (min,max) over which the final field is defined
    
    yFRAnge: tuple (meters)
        The y-range (min,max) over which the final field is defined


    Returns:
    ------------
    xF,yF: 1d numpy arrays (meters)
        Defining the spatial extent over which the field is calculated in the diffracted plane.

    diffractedField: 2d numpy array (complex E-Field)
        Representing the complex electric field profile in the diffracted plane.
   
    """
    k = 2*np.pi/wavelength
    X,Y = np.meshgrid(x,y)

    preFactor = np.exp( 1j*k*z ) *np.exp(1j*k*(X**2 + Y**2)/(2*z))/(1j*wavelength*z)

    kxRange = ( xFRange[0]/wavelength/z*2*np.pi , xFRange[1]/wavelength/z*2*np.pi )
    kyRange = ( yFRange[0]/wavelength/z*2*np.pi , yFRange[1]/wavelength/z*2*np.pi )

    (k_x,k_y,F) = zoomFourierTransform2D(x,y,field*preFactor,kxRange,kyRange)
    #(k_x,k_y,F) = fourierTransform2D(x,y,field*preFactor)
    
    xF = k_x*wavelength*z/2/np.pi
    yF = k_y*wavelength*z/2/np.pi


    (XF,YF) = np.meshgrid(xF,yF)
    
    postFactor = np.exp( 1j*k/z * (XF**2 + YF**2) ) 


    diffractedField= F*postFactor

    return (xF,yF,diffractedField)


    