#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:34:56 2021

@author: dejan
"""
import numpy as np
from skimage import transform, io


def pV(x: np.ndarray,
       h: float, x0: float = None, w: float = None, factor: float = 0.5):
    """Create a pseudo-Voigt profile.

    Parameters
    ----------
    x : 1D ndarray
        Independent variable (Raman shift for ex.)
    h : float
        The height of the peak
    x0 : float
        The position of the peak on the x-axis.
        Default value is at the middle of the x
    w : float
        FWHM - The width
        Default value is 1/3 of the x
    factor : float
        The ratio of Gauss vs Lorentz in the peak
        Default value is 0.5

    Returns
    -------
    y : np.ndarray :
        1D-array of the same length as the input x-array
    ***************************

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(150, 1300, 1015)
    >>> plt.plot(x, pV(x, 200))
    """

    def Gauss(x, w):
        return((2/w) * np.sqrt(np.log(2)/np.pi) * np.exp(
            -(4*np.log(2)/w**2) * (x - x0)**2))

    def Lorentz(x, w):
        return((1/np.pi)*(w/2) / (
            (x - x0)**2 + (w/2)**2))

    if x0 is None:
        x0 = x[int(len(x)/2)]
    if w is None:
        w = (x.max() - x.min()) / 3

    intensity = h * np.pi * (w/2) /\
        (1 + factor * (np.sqrt(np.pi*np.log(2)) - 1))

    return(intensity * (factor * Gauss(x, w)
                        + (1-factor) * Lorentz(x, w)))


def multi_pV(x, params, peak_function=pV):
    """Create the spectra as the sum of the pseudo-Voigt peaks.

    You need to provide the independent variable `x`
    and a set of parameters for each peak.
    (one sublist for each Pseudo-Voigt peak).

    Parameters
    ----------
    x : np.ndarray
        1D ndarray - independent variable.
    *params : list[list[float]]
        The list of lists containing the peak parameters. For each infividual
        peak to be created there should be a sublist of parameters to be
        passed to the pV function. So that `params` list finally contains
        one of these sublists for each Pseudo-Voigt peak to be created.
        Look in the docstring of pV function for more info
        on what to put in each sublist.
        (h, x0, w, G/L) - only the first parameter is mandatory.

    Returns
    -------
    y : np.ndarray
        1D ndarray of the same length as the input x-array

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(150, 1300, 1015) # Create 1015 equally spaced points
    >>> mpar = [[40, 220, 100], [122, 440, 80], [164, 550, 160], [40, 480, 340]]
    >>> plt.plot(x, multi_pV(x, mpar))
    """
    result = np.zeros_like(x, dtype=np.float)
    for pp in params:
        result += peak_function(x, *pp)  # h, x0, w, r
    return result


def create_multiple_spectra(x: np.ndarray, initial_peak_params: list,
                            defaults=None, N=10000, noise: float = 0.02,
                            spectrum_function=multi_pV,
                            noise_bias='linea', funny_peak='random'):
    """Create N different spectra using mutli_pV function.

    Parameters
    ----------
    x : np.ndarray
        1D ndarray - independent variable
    initial_peak_params: list[float]
        The list of sublists containing individual peak parameters as
        demanded by the `spectrum_function`.
    defaults: list, optional
        Default params for the sublists where not all params are set.
        The function will try to come up with something if the defaults
        are not provided.
    N : int, optional
        The number of spectra to create. Defaults to 1024 (32x32 :)
    noise : float
        Noisiness and how much you want the spectra to differ between them.
    spectrum_function : function
        The default is multi_pV.
        You should be able to provide something else, but this is not yet
        tested.
    noise_bias: None or 'smiley' or 'linea'
        Default is 'linea'.
        The underlaying pattern of the differences between the spectra.
    funny_peak: int or list of ints or 'random' or 'all'
        Only applicable if `noise_bias` is 'smiley'.
        Designates the peak on which you want the bias to appear.
        If 'random', one peak is chosen randomly.

    Returns
    -------
    y : np.ndarray :
        2D ndarray of the shape (N, len(x))

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(150, 1300, 1015) # Create 1015 equally spaced points
    >>> mpar = [[40, 220, 100], [122, 440, 80], [164, 550, 160], [40, 480, 340]]
    >>> my_spectra = create_multiple_spectra(x, mpar)
    """

    def binarization_load(f, shape=(132, 132)):
        """May be used if "linea" mode is active."""
        im = io.imread(f, as_gray=True)
        return transform.resize(im, shape, anti_aliasing=True)

    # We need to make sure that all the sublists are of the same length.
    # If that is not the case, we need to fill the sublists with the default
    # values.
    if defaults is None:
        defaults = [np.median([pp[0] for pp in initial_peak_params]),
                    np.median(x),
                    np.ptp(x)/20,
                    0.5]
    for i, par in enumerate(initial_peak_params):
        while len(par) < len(defaults):
            initial_peak_params[i].append(defaults[len(par)])
    n_peaks = len(initial_peak_params)  # Number of peaks
    ponderation = 1 + (np.random.rand(N, len(defaults), 1) - 0.5) * noise
    peaks_params = ponderation * np.asarray(initial_peak_params)
    # -------- The funny part ----------------------------------
    if noise_bias == 'smiley':
        smile = io.imread('./misc/bibi.jpg')
        x_dim = int(np.sqrt(N))
        y_dim = N//x_dim
        print(f"You'll end up with {x_dim}*{y_dim} = {x_dim*y_dim} points"
              f"instead of initial {N}")
        N = x_dim * y_dim
        smile_resized = transform.resize(smile, (x_dim, y_dim))
        noise_bias = smile_resized.ravel()
        if funny_peak == 'random':
            funny_peak = np.random.randint(0, n_peaks+1)
        elif funny_peak == 'all':
            funny_peak = list(range(n_peaks))
        peaks_params[:, funny_peak, 0] *= noise_bias
    elif noise_bias == 'linea':
        x_dim = int(np.sqrt(N))
        y_dim = N//x_dim
        images = './misc/linea/*.jpg'
        coll_all = io.ImageCollection(images, load_func=binarization_load,
                                      shape=(x_dim, y_dim))
        print(f"You'll end up with {x_dim}*{y_dim} = {x_dim*y_dim} points"
              f"instead of initial {N}")
        N = x_dim * y_dim
    # -------- The End of the funny part ------------------------
    additive_noise = peaks_params[:, :, 0].mean() *\
        (0.5+np.random.rand(len(x))) / 5
    spectra = np.asarray(
        [multi_pV(x, peaks_params[i]) +
         additive_noise[np.random.permutation(len(x))]
         for i in range(N)])
    if isinstance(noise_bias, str) and noise_bias == 'linea':
        noise_bias = coll_all.concatenate().reshape(110, -1)
        spectra[:, -110:] *= noise_bias.T

    return spectra.reshape(N, -1)
