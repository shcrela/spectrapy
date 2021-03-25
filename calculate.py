#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:50:13 2021

@author: dejan
"""


def find_barycentre(x, y, method="trapz_minimize"):
    '''Calculates the index of the barycentre value.
        Parameters:
        ----------
        x:1D ndarray: ndarray containing your raman shifts
        y:1D ndarray: Ndarray containing your intensity (counts) values
        method:string: only "trapz_minimize" for now
        Returns:
        ---------
        (x_value, y_value): the coordinates of the barycentre
        '''
    assert(method in ['trapz_minimize'])#, 'sum_minimize', 'trapz_list'])
    #razlika = np.asarray(np.diff(x, append=x[-1]+x[-1]-x[-2]), dtype=np.float16)
    #assert(np.all(razlika/razlika[np.random.randint(len(x))] == np.ones_like(x))),\
    #"your points are not equidistant"
    half = np.trapz(y, x=x)/2
    #from scipy.interpolate import interp1d
    #xx=np.linspace(x.min(), x.max(), 2*len(x))
    #f = interp1d(x, y, kind='quadratic')
    #yy = f(xx)
    if method in 'trapz_minimize':
        def find_y(Y0, xx=x, yy=y, method=method):
            '''Internal function to minimize
            depending on the method chosen'''
            # Calculate the area of the curve above the Y0 value:
            part_up = np.trapz(yy[yy>=Y0]-Y0, x=xx[yy>=Y0])
            # Calculate the area below Y0:
            part_down = np.trapz(yy[yy<=Y0], x=xx[yy<=Y0])
            # for the two parts to be the same
            to_minimize_ud = np.abs(part_up - part_down)
            # fto make the other part be close to half
            to_minimize_uh = np.abs(part_up - half)
            # to make the other part be close to half
            to_minimize_dh = np.abs(part_down - half)
            return to_minimize_ud**2+to_minimize_uh+to_minimize_dh

        def find_x(X0, xx=x, yy=y, method=method):
            part_left = np.trapz(yy[xx<=X0], x=xx[xx<=X0])
            part_right = np.trapz(yy[xx>=X0], x=xx[xx>=X0])
            to_minimize_lr = np.abs(part_left - part_right)
            to_minimize_lh = np.abs(part_left - half)
            to_minimize_rh = np.abs(part_right - half)
            return to_minimize_lr**2+to_minimize_lh+to_minimize_rh

        minimized_y = minimize_scalar(find_y, method='Bounded',
                                    bounds=(np.quantile(y, 0.01),
                                            np.quantile(y, 0.99)))
        minimized_x = minimize_scalar(find_x, method='Bounded',
                                    bounds=(np.quantile(x, 0.01),
                                            np.quantile(x, 0.99)))
        y_value = minimized_y.x
        x_value = minimized_x.x

    elif method == "list_minimize":
        ys = np.sort(yy)
        z2 = np.asarray(
            [np.abs(np.trapz(yy[yy<=y_val], x=xx[yy<=y_val]) -\
                    np.trapz(yy[yy>=y_val]-y_val, x=xx[yy>=y_val]))\
             for y_val in ys])
        y_value = ys[np.argmin(z2)]
        x_ind = np.argmin(np.abs(np.cumsum(yy) - np.sum(yy)/2)) + 1
        x_value = xx[x_ind]

    return x_value, y_value


def rolling_median(arr, w_size, ax=0, mode='nearest', *args):
    '''Calculates the rolling median of an array
    along the given axis on the given window size.
    Parameters:
    -------------
        arr:ndarray: input array
        w_size:int: the window size
                    (should be less then the dimension along the given axis)
        ax:int: the axis along which to calculate the rolling median
        mode:str: to choose from ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
        see the docstring of ndimage.median_filter for details
    Returns:
    ------------
        ndarray of same shape as the input array'''
    shape = np.ones(np.ndim(arr), dtype=int)
    shape[ax] = w_size
    return median_filter(arr, size=shape, mode=mode, *args)



def baseline_als(y, lam=1e5, p=5e-5, niter=12):
    '''Adapted from:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    To get the feel on how the algorithm works, you can think of it as
    if the rolling ball which comes from beneath the spectrum and thus sets
    the baseline.

    Then, to follow the image, schematic explanaton of the params would be:

    Params:
    ----------
        y:          1D or 2D ndarray: the spectra on which to find the baseline

        lam:number: Can be viewed as the radius of the ball.
                    As a rule of thumb, this value should be around the
                    twice the width of the broadest feature you want to keep
                    (width is to be measured in number of points, since
                    for the moment no x values are taken into accound
                    in this algorithm)

        p:number:   Can be viewed as the measure of how much the ball
                    can penetrate into the spectra from below

        niter:int:  number of iterations
                   (the resulting baseline should stabilize after
                    some number of iterations)

    Returns:
    -----------
        b_line:ndarray: the baseline (same shape as y)

    Note:
    ----------
        It takes around 2-3 sec per 1000 spectra with 10 iterations
        on i7 4cores(8threads) @1,9GHz

    '''
    def _one_bl(yi, lam=lam, p=p, niter=niter, z=None):
        if z is None:
            L = yi.shape[-1]
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
            w = np.ones(L)
            W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w) # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = sparse.linalg.spsolve(Z, w*yi)
            w = p * (yi > z) + (1-p) * (yi < z)
        return z

    if y.ndim == 1:
        b_line = _one_bl(y)
    elif y.ndim == 2:
        b_line = np.asarray(Parallel(n_jobs=-1)(delayed(_one_bl)(y[i])
                                                for i in range(y.shape[0])))
    else:
        warn("This only works for 1D or 2D arrays")

    return b_line