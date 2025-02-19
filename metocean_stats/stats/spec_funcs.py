import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import CubicSpline,RegularGridInterpolator
from scipy.integrate import simpson

def jonswap(f,hs,tp,gamma='fit', sigma_low=.07, sigma_high=.09):
    """
    Purpose:
        To determine spectral density based on the JONSWAP spectrum 

    Input:
        hs  - Significant wave height
        tp  - Spectral peak period
        f   - array of Wave frequency

    Output:
        sf  - Spectral density
    """
    g = 9.82
    fp = 1/tp
    if gamma == 'fit':
       gamma = min(np.exp(3.484*(1-0.1975*(0.036-0.0056*tp/np.sqrt(hs))*(tp**4)/hs**2)),5) # see MET-report_03-2021.pdf, max value should not exceed 5
    else:
       pass
    
    #print('gamma-JONSWAP is',gamma)
    alpha  = 5.061*(hs**2/tp**4)*(1-0.287*np.log(gamma)) # see MET-report_03-2021.pdf
    E_pm = alpha*(g**2)*((2*np.pi)**-4)*f**-5*np.exp((-5/4)*((fp/f)**4))
    sigma = np.ones(f.shape)*sigma_low
    sigma[f > 1./tp] = sigma_high
    E_js = E_pm*gamma**np.exp(-0.5*(((f/fp)-1)/sigma)**2)   # see MET-report_03-2021.pdf
    sf = np.nan_to_num(E_js)
    return sf

def velocity_spectrum(f, S_w, depth, ref_depth):
    g = 9.82
    h = depth 
    z = ref_depth
    #k = (1/g)*(2*np.pi/f)**2
    k, ik = waveno(t=1/f,h =depth) 
    k = np.nan_to_num(k)
    G = 2*np.pi *f* np.cosh(k*(depth-ref_depth))/np.sinh(k*ref_depth)
    G = 2*np.pi*f*np.exp(-k*z)*(1+np.exp(-2*(k*h-k*z)))/(1.-np.exp(-2*k*h))
    G = np.nan_to_num(G)
    S_uu = S_w*G**2
    return S_uu



def torsethaugen(f, hs, tp):
    """
    Purpose:
        To determine spectral density based on the Torsethaugen double peaked spectrum 

    Input:
        hs  - Significant wave height
        tp  - Spectral peak period
        f   - array of Wave frequency

    Output:
        sf  - Spectral density
    """
    # Constants
    pi = np.pi
    g = 9.81
    af, ae, au = 6.6, 2.0, 25.0
    a10, a1, rkg = 0.7, 0.5, 35.0
    b1, a20, a2, a3 = 2.0, 0.6, 0.3, 6.0
    g0 = 3.26

    tpf = af * hs ** (1.0 / 3.0)
    tl = ae * hs ** (1.0 / 2.0)
    el = (tpf - tp) / (tpf - tl)

    if tp <= tpf:
        rw = (1.0 - a10) * np.exp(-(el / a1) ** 2) + a10
        hw1 = rw * hs
        tpw1 = tp
        sp = (2.0 * pi / g) * hw1 / tpw1 ** 2
        gam1 = max(1.0, rkg * sp ** (6.0 / 7.0))

        hw2 = np.sqrt(1.0 - rw ** 2) * hs
        tpw2 = tpf + b1
        h1, tp1 = hw1, tpw1
        h2, tp2 = hw2, tpw2
    else:
        tu = au
        eu = (tp - tpf) / (tu - tpf)
        rs = (1.0 - a20) * np.exp(-(eu / a2) ** 2) + a20
        hs1 = rs * hs
        tps1 = tp
        sf = (2.0 * pi / g) * hs / tpf ** 2
        gam1 = max(1.0, rkg * sf ** (6.0 / 7.0) * (1.0 + a3 * eu))
        hs2 = np.sqrt(1.0 - rs ** 2) * hs
        tps2 = af * hs2 ** (1.0 / 3.0)
        h1, tp1 = hs1, tps1
        h2, tp2 = hs2, tps2

    e1 = (1.0 / 16.0) * (h1 ** 2) * tp1
    e2 = (1.0 / 16.0) * (h2 ** 2) * tp2
    ag = (1.0 + 1.1 * np.log(gam1) ** 1.19) / gam1

    f1n, f2n = f * tp1, f * tp2
    
    sigma1 = np.where(f1n > 1.0, 0.09, 0.07)
    
    fnc1 = f1n ** (-4) * np.exp(-f1n ** (-4))
    fnc2 = gam1 ** (np.exp(-1.0 / (2.0 * sigma1 ** 2) * (f1n - 1.0) ** 2))
    s1 = g0 * ag * fnc1 * fnc2

    fnc3 = f2n ** (-4) * np.exp(-f2n ** (-4))
    s2 = g0 * fnc3

    sf = e1 * s1 + e2 * s2
    return np.nan_to_num(sf)


def waveno(t, h):
    """
    Purpose:
        To compute wave number

    Input:
        t  - Wave period
        h  - Water depth (can be an array or a single value)

    Output:
        k - Wave number 
        nier - Negative depth values: nier = 1
    """
    # Set value of constants
    g = 9.82
    A = [0.66667, 0.35550, 0.16084, 0.06320, 0.02174, 0.00654, 0.00171, 0.00039, 0.00011]

    # Initialize output
    nier = 0
    sigma = 2.0 * np.pi / t

    # Function to compute wave number for a single depth
    def compute_k(depth):
        nonlocal nier
        b = g * depth
        if b < 0:
            nier = 1
            return 0.0
        y = sigma * sigma * depth / g
        x = A[4] + y * (A[5] + y * (A[6] + y * (A[7] + y * A[8])))
        x = 1.0 + y * (A[0] + y * (A[1] + y * (A[2] + y * (A[3] + y * x))))
        c = np.sqrt(b / (y + 1.0 / x))
        return sigma / c

    # Check if h is a single value or a list
    if isinstance(h, (int, float)):
        k = compute_k(h)
    else:
        k = [compute_k(depth) for depth in h]

    return k, nier

def _interpolate_cubic(fp,n):
    '''
    Cubic spline interpolation within the range of fp, for resampling.

    Parameters
    ----------
    fp : np.ndarray
        Values of the function
    n : int
        New sampling resolution.
    '''
    l = len(fp)
    x = np.linspace(0,l-1,n)
    xp = np.arange(l)
    spl = CubicSpline(xp,fp)
    return spl(x)

def interpolate_2D_spec(spec  : np.ndarray,
                        freq0 : np.ndarray,
                        dir0  : np.ndarray,
                        freq1 : np.ndarray,
                        dir1  : np.ndarray,
                        method: str="linear"
                        ) -> np.ndarray:
    '''
    Interpolate 2D wave spectra from fre0 and dir0 to freq1 and dir1.
    
    Parameters
    ---------
    spec : np.ndarray
        N-D array of spectra, must have dimensions [..., frequencies, directions].
    freq0 : np.ndarray
        Array of frequencies.
    dir0 : np.ndarray
        Array of directions.
    freq1 : np.ndarray
        Array of new frequencies.
    dir1 : np.ndarray
        Array of new directions.
    method : str
        The interpolation method used by scipy.interpolate.RegularGridInterpolator(),
        e.g. "nearest", "linear", "cubic", "quintic".
        
    Returns
    -------
    spec : np.ndarray
        The interpolated spectra.
    '''
    # Sort on directions, required for interpolation.
    sorted_indices = np.argsort(dir0)
    dir0 = dir0[sorted_indices]
    spec = spec[...,sorted_indices]

    # Create current and new interpolation points.
    points = tuple(np.arange(s) for s in spec.shape[:-2]) + (freq0,dir0)
    coords = tuple(np.arange(s) for s in spec.shape[:-2]) + (freq1,dir1)
    reorder = tuple(np.arange(1,len(coords)+1))+(0,)
    coords = np.transpose(np.meshgrid(*coords,indexing="ij"),reorder)

    # Define interpolator and interpolate.
    grid_interp = RegularGridInterpolator(points=points,values=spec,fill_value=None,bounds_error=False)
    return grid_interp(coords,method=method)

def scale_2D_spec(  spec:np.ndarray,
                    frequencies : np.ndarray,
                    directions  : np.ndarray,
                    new_frequencies: np.ndarray | int = 20,
                    new_directions: np.ndarray | int = 20,
                    method="linear"
                    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Interpolate wave spectra to a new set of specific frequencies and directions.
    
    Parameters
    ----------
    spec : np.ndarray
        Array of spectra, with last two dimensions corresponding to frequencies and directions in order.
    frequences : np.ndarray
        Array of frequency values, corresponding to the second-last dimension of spec.
    directions : np.ndarray
        Array of direction values (degrees), corresponding to the last dimension of spec.
    new_frequencies : int or np.ndarray
        New frequency values, or an integer for number of (interpolated) frequency values.
    new_directions : int or np.ndarray
        New directions, or an integer for number of (uniformly distributed) new directions.
    method : str
        Any interpolation method allowed by scipy's RegularGridInterpolator, such as 
        "nearest", "linear", "cubic", "quintic".

    Returns
    --------
    new_spec : np.ndarray 
        Interpolated spectra, array with same shape as input spectra except the last two dimensions.
    new_frequencies : np.ndarray
        The new set of frequencies
    new_directions : np.ndarray
        The new set of directions.
    '''

    new_frequencies = np.array(new_frequencies)
    new_directions = np.array(new_directions)
    
    # Create freqs and dirs through interpolation if not supplied directly
    if new_frequencies.size == 1:
        new_frequencies = _interpolate_cubic(frequencies,new_frequencies)
    if new_directions.size == 1:
        new_directions = np.linspace(0,360,new_directions,endpoint=False)

    spec = interpolate_2D_spec(spec,frequencies,directions,new_frequencies,new_directions,method=method)
    return spec, new_frequencies, new_directions

def interpolate_dataarray_spec( spec: xr.DataArray,
                                new_frequencies: np.ndarray | int = 20,
                                new_directions: np.ndarray | int = 20,
                                method="linear"
                                ):
    '''
    Interpolate 2D wave spectra to a new shape.
    The last two dimensions of spec must represent frequencies and directions.
    This is just a wrapper for scale_2D_spec, to keep track of the dataarray metadata.
    
    Parameters
    ---------
    spec : xr.DataArray
        Array of spectra. Must have dimensions [..., frequencies, directions].
    new_frequencies : xr.DataArray or np.ndarray or int
        Either an array of new frequences, or an integer for the number of new frequencies.
        If integer, new frequencies will be created with cubic interpolation.
    new_directions : xr.DataArray or np.ndarray or int
        Either an array of new directions, or an integer for the number of new directions.
        If integer, new directions will be created with linear interpolation.
    method : str
        The interpolation method used by scipy.interpolate.RegularGridInterpolator(),
        e.g. "nearest", "linear", "cubic", "quintic".
    
    Returns
    -------
    spec : xr.DataArray
        The 2D-interpolated spectra.
    '''

    # Extract dimension labels and coordinate arrays from spec.
    spec_coords = spec.coords
    spec_dims = list(spec.dims)
    freq_var = spec_dims[-2]
    dir_var = spec_dims[-1]
    free_dims = spec_dims[:-2]

    frequencies = spec_coords[freq_var]
    directions = spec_coords[dir_var]

    new_spec, new_frequencies, new_directions = scale_2D_spec(
        spec.data,frequencies,directions,new_frequencies,new_directions,method=method)
    
    new_coordinates = {k:spec_coords[k] for k in free_dims}
    new_coordinates[freq_var] = new_frequencies
    new_coordinates[dir_var] = new_directions
    return xr.DataArray(new_spec,new_coordinates)

def _reshape_spectra(spec, frequencies, directions):
    '''
    Standardize format of spectra, frequencies and directions for further processing.

    Parameters
    ----------
    spec : np.ndarray or pd.DataFrame or xr.DataArray
        Array of spectra.
    frequencies : np.ndarray
        List of freqency values corresponding to the second-last dimension of the spectra.
    directions : np.ndarray
        List of directions corresponding to the last dimension of the spectra.

    Returns
    -------
    np.ndarray
        Spectra shape [.., frequencies, dimensions]
    np.ndarray
        Frequencies
    np.ndarray
        Directions
    '''
    # Make sure all arrays are numpy.
    spec = np.array(spec)
    frequencies = np.array(frequencies)
    directions = np.array(directions)

    # Check if spec values and shape are OK
    if np.any(spec < 0):
        print("Warning: negative spectra values set to 0")
        spec = np.clip(spec, a_min=0, a_max=None)

    flat_check = (len(spec.shape)<2)
    if not flat_check:
        freq_check = (len(frequencies) != spec.shape[-2])
        dir_check = (len(directions) != spec.shape[-1])
    if flat_check or freq_check or dir_check:
        try:
            spec = spec.reshape(spec.shape[:-1]+(len(frequencies),len(directions)))
        except:
            raise IndexError("Spec shape does not match frequencies and directions.")

    return spec, frequencies, directions
    
def integrated_parameters(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray,
    upsample: int = 1000) -> dict:
    """
    Calculate the integrated parameters of a 2D wave spectrum, 
    or some array/list of spectra. Uses simpsons integration rule.

    Implemented: Hs, peak dir, peak freq.
    
    Parameters
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
    upsample: int (optional, recommended)
        Upsample frequencies and directions by cubic spline interpolation
        to increase the resolution of peak_dir and peak_freq.
        Does not affect other integrated parameters.
        
    Returns
    -------
    spec_parameters : dict[str, np.ndarray]
        A dict with keys Hs, peak_freq, peak_dir, and values are arrays
        of the integrated parameter.
    """

    spec, frequencies, directions = _reshape_spectra(spec, frequencies, directions)
    
    # # Use argmax to find indices of largest value of each spectrum.
    # peak_dir_freq = np.array([np.unravel_index(s.argmax(),s.shape) 
    #     for s in spec.reshape(-1,len(frequencies),len(directions))])
    # peak_dir_freq = peak_dir_freq.reshape(spec.shape[:-2]+(2,))
    # peak_freq = frequencies[peak_dir_freq[...,0]]
    # peak_dir = directions[peak_dir_freq[...,1]]
    peak_freq, peak_dir = peak_freq_dir(spec,frequencies,directions,upsample=upsample)
    
    # Integration requires radians
    if np.max(directions) > 2*np.pi: 
        directions = np.deg2rad(directions)
    
    # Sort on direction before integration
    sorted_indices = np.argsort(directions)
    directions = directions[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Wraparound correction
    directions = np.concatenate([directions-directions[0],[2*np.pi]])
    spec = np.concatenate([spec,spec[...,:1]],axis=-1)

    # Integration with simpson's rule
    S_f = simpson(spec, x=directions)
    m0 = simpson(S_f, x=frequencies)
    Hs = 4 * np.sqrt(m0)

    spec_parameters = {
        "Hs":       Hs,
        "peak_freq":peak_freq,
        "peak_dir": peak_dir,
        "peak_period":1/peak_freq
    }

    return spec_parameters

def peak_freq_dir(spec:np.ndarray,
                  frequencies:np.ndarray,
                  directions:np.ndarray,
                  upsample:int=1000):
    '''
    Calculate peak frequency and direction of spectra.

    Parameters
    ----------
    spec : np.ndarray
        Some array of spectra.
    frequencies : np.ndarray
        Frequencies corresponding to the second-last dimension of the spectra.
    directions : np.ndarray
        Directions corresponding to the last dimension of the spectra.
    upsample : int
        Upsample the frequency spectra and direction spectra via 
        cubic splines, for higher resolution of the peak values.

    Returns
    -------
    peak_freq : np.ndarray
        Peak frequency, shape corresponding to the number of input spectra.
    peak_dir : np.ndarray
        Peak directions, shape corresponding to the number of input spectra.

    Notes
    ------
    This function first integrates directions to get frequency peak,
    then integrates over frequencies to get direction peak. In some rare
    cases will this be different to the overall 2D spectra peaks,
    and it is much more computationally efficient due to 1D interpolation.
    '''
    
    freq_spec, frequencies = frequency_spectra(spec,frequencies,directions)
    dir_spec, directions = direction_spectra(spec,frequencies,directions)
    
    if upsample:
        xd = np.linspace(0,360,upsample,endpoint=False)
        dir_spec = np.concat([dir_spec,dir_spec[...,:1]],axis=-1)
        directions = np.concat([directions-directions[0],[360]])

        xf = _interpolate_cubic(frequencies,upsample)

        freq_spec = CubicSpline(frequencies,freq_spec,axis=-1)(xf) 
        dir_spec =  CubicSpline(directions, dir_spec, axis=-1,bc_type='periodic')(xd) 

        peak_freq = xf[freq_spec.argmax(axis=-1)]
        peak_dir = xd[dir_spec.argmax(axis=-1)]

    else:
        peak_freq = frequencies[freq_spec.argmax(axis=-1)]
        peak_dir = directions[dir_spec.argmax(axis=-1)]

    return peak_freq, peak_dir

def frequency_spectra(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray
    ) -> np.ndarray|xr.DataArray:
    '''
    Get frequency spectra by integrating over directions.
    
    Parameters
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
        
    Returns
    -------
    1D_spec : np.ndarray or xr.DataArray
        Array of 1D spectra.
    '''

    spec, frequencies, directions = _reshape_spectra(spec,frequencies,directions)
    
    # Integration requires radians
    if np.max(directions) > 2*np.pi: 
        directions = np.deg2rad(directions)
    
    # Sort on direction before integration
    sorted_indices = np.argsort(directions)
    directions = directions[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Wraparound correction
    directions = np.concatenate([directions-directions[0],[2*np.pi]])
    spec = np.concatenate([spec,spec[...,:1]],axis=-1)

    # Integration with simpson's rule
    S_f = simpson(spec, x=directions)
    
    return S_f, frequencies

def direction_spectra(
    spec:       np.ndarray|xr.DataArray, 
    frequencies:np.ndarray|xr.DataArray, 
    directions: np.ndarray|xr.DataArray
    ) -> np.ndarray|xr.DataArray:
    '''
    Get direction spectra by integrating frequencies.
    
    Parameters
    ---------
    spec : np.ndarray or xr.DataArray
        An array of spectra. The shape must be either 
        [..., frequencies, directions] or [..., frequencies*directions].
    frequencies : np.ndarray or xr.DataArray
        Array of spectra frequencies.
    directions: np.ndarray or xr.DataArray
        Array of spectra directions.
        
    Returns
    -------
    1D_spec : np.ndarray or xr.DataArray
        Array of 1D spectra.
    '''

    spec, frequencies, directions = _reshape_spectra(spec,frequencies,directions)
    
    # Sort on direction before integration
    sorted_indices = np.argsort(directions)
    directions = directions[sorted_indices]
    spec = spec[...,sorted_indices]
    
    # Integration with simpson's rule
    S_d = simpson(spec, x=frequencies, axis=-2)
    
    return S_d, directions
