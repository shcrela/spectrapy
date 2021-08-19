#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:54:49 2021

@author: dejan
"""
import numpy as np
import scipy.ndimage as nd
from visualize import NavigationButtons
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import scipy
from scipy.ndimage import median_filter
from tqdm import tqdm
from timeit import default_timer as time

#sns.set()
# %%
# =============================================================================
#                                 CR correction...
# =============================================================================
def remove_cosmic_rays(spectra, sigma=None, sensitivity=1):
    '''Attention: The returned spectra is normalized over 1+area
    This method relies on neighbours to identify the cosmic ray candidates
    as parts of spectra too different from the neighbours.

    Parameters:
    ------------------
    spectra: ndarray
        IMPORTANT: if the spectra are comming from the map, then the spectra
        has to be properly shaped.

        >>> spectra.shape
        (nx, ny, n_shifts)

        Otherwise, if the spectra are from time series, leave them
        in their shape.

        >>> spectra.shape
        (n_measurements, n_shifts)

    sigma: ndarray
        The Raman shifts
        default value is None.
        If left as default None, the function will use np.arange(n_shifts)

    sensitivity: float > 0
        default value is 1
        bigger -> more sensitive (more Cosmic rays will be detected).\n
        smaller -> less sensitive (it will detect only the most prominent CRs).

    Returns:
    -----------------
    cleaned spectra : ndarray
        The array of the same shape as the imput spectra, only normalized and
        (hopefully) with cosmic rays removed.

    CR_cand_ind: ndarray
        ndarray containing the indices of the initial spectra where the
        cosmic rays were detected.
    '''
    nedim = np.ndim(spectra)
    if nedim == 3:
        ny, nx, n_shifts = spectra.shape
    elif nedim == 2:
        nx = 1
        ny, n_shifts = spectra.shape
    else :
        raise ValueError("Check your input spectra's shape.")
    if sigma is None:
        sigma = np.arange(n_shifts)
    if nx > 4:
        # construct the footprint pointing to the pixels surrounding any given pixel:
        tt = np.zeros((5,nx,1))
        tt[2,:5] = tt[:,2] = tt[1,1] = tt[1,3] = tt[3,1] = tt[3,3] = 1
        tt[2,2] = 0
    elif nx == 1:
        tt = np.ones((5,1))
    else:
        raise ValueError(f'The width "nx" should be > 3, and yours is {nx}.')
    kkk = tt.ravel()[:, np.newaxis]
    spectra = spectra.reshape(nx*ny, n_shifts)
    scaling_koeff = 1 + np.abs(np.trapz(spectra, x=sigma,
                                        axis=-1)[:, np.newaxis])
    normalized_spectra = np.copy(spectra / scaling_koeff)
    # each pixel has the median value of its surrounding neighbours:
    median_spectra3 = np.copy(nd.median_filter(normalized_spectra,
                                            footprint=kkk, mode='nearest'))

    # I will only take into account the positive values (CR):
    coarsing_diff = normalized_spectra - median_spectra3

    # For each spectra, find the zone where the difference between the given
    # spectra and its' neighbours is the greatest.
    basic_candidates = np.nonzero(coarsing_diff >\
                                  20*np.std(coarsing_diff) / sensitivity)
    sind = basic_candidates[0] # the spectra containing very bad neighbours
    # rind = basic_candidates[1] # each element from the "very bad neighbour"
    if len(sind) > 0:
        # =====================================================================
        #               We want to extend the "very bad neighbour" label
        #           to ext_size adjecent family members in each such spectra:
        # =====================================================================
        candidates_mask = np.zeros_like(normalized_spectra, dtype=bool)
        candidates_mask[basic_candidates] = True
        largest_cr = 5*np.sum(candidates_mask, axis=-1)
        extended_mask = np.copy(candidates_mask)
        for i, zone in enumerate(candidates_mask):
            if largest_cr[i] > 0:
                extended_mask[i] = nd.binary_dilation(zone,
                                        structure=np.ones((largest_cr[i]),))
    cleaned_spectra = np.copy(normalized_spectra)
    CR_cand_ind = np.unique(sind)

    # If one is willing to neglect the probability of the situation where
    # one spectra has its' cosmic ray at the rightmost pixel, and the spectrum
    # that follows it has the cosmic ray at the leftmost pixel, then we can
    # label the zones to be replaced like this:
    labels = nd.label(extended_mask.ravel(),
                      structure=np.ones(3))[0].reshape(extended_mask.shape)
    # `labels` label each zone to be replaced differently

    # These are the points where the zone to be replaced starts
    # (some spectra may have multiple such zones)
    stiching_points = np.nonzero(np.diff(extended_mask, n=1, axis=-1,
                                         append=0)==1)
    koefs = normalized_spectra[stiching_points]/\
        median_spectra3[stiching_points]

    for i in np.arange(1, len(koefs)+1):
        cleaned_spectra[labels==i] = median_spectra3[labels==i]*koefs[i-1]
    # Ilustration:
    _ss = np.stack((normalized_spectra[CR_cand_ind],
                    cleaned_spectra[CR_cand_ind]),
                   axis=-1)
    title = [f"indice={i}" for i in CR_cand_ind]
    check_CR_candidates = NavigationButtons(sigma, _ss,
                                            autoscale_y=True,
                                            title=title,
                                            label=['normalized spectra',
                                                   'median correction']);
    check_CR_candidates.figr.suptitle("Cosmic Rays Correction")

    return cleaned_spectra.reshape((ny, nx, n_shifts)), CR_cand_ind


#%%
def remove_CRs(mock_sp3, sigma_kept, _n_x=0, _n_y=0, **initialization):
    # a bit higher then median, or the area:
    scaling_koeff = np.trapz(mock_sp3, x=sigma_kept, axis=-1)[:, np.newaxis]
    mock_sp3 /= np.abs(scaling_koeff)
    normalized_spectra = np.copy(mock_sp3)
    # construct the footprint pointing to the pixels surrounding any given pixel:
    kkk = np.zeros((2*(_n_x+1) + 1, 1))
    # does this value change anything?
    kkk[[0, 1, 2, _n_x-1, _n_x+1, -3, -2, -1]] = 1

    # each pixel has the median value of its surrounding neighbours:
    median_spectra3 = median_filter(mock_sp3, footprint=kkk)

    # I will only take into account the positive values (CR):
    coarsing_diff = (mock_sp3 - median_spectra3)

    # find the highest differences between the spectra and its neighbours:
    bad_neighbour = np.quantile(coarsing_diff, 0.99, axis=-1)
    # The find the spectra where the bad neighbour is very bad:
    # The "very bad" limit is set here at 30*standard deviation (why not?):
    basic_candidates = np.nonzero(coarsing_diff > 40*np.std(bad_neighbour))
    sind = basic_candidates[0]  # the spectra containing very bad neighbours
    rind = basic_candidates[1]  # each element from the "very bad neighbour"
    if len(sind) > 0:
        # =====================================================================
        #               We want to extend the "very bad neighbour" label
        #           to ext_size adjecent family members in each such spectra:
        # =====================================================================
        npix = len(sigma)
        ext_size = int(npix/50)
        if ext_size % 2 != 1:
            ext_size += 1
        extended_sind = np.stack((sind, )*ext_size, axis=-1).reshape(
            len(sind)*ext_size,)
        rind_stack = tuple()
        for ii in np.arange(-(ext_size//2), ext_size//2+1):
            rind_stack += (rind + ii, )
        extended_rind = np.stack(rind_stack, axis=-1).reshape(
            len(rind)*ext_size,)
        # The mirror approach for family members close to the border:
        extended_rind[np.nonzero(extended_rind < 0)] =\
            -extended_rind[np.nonzero(extended_rind < 0)]
        extended_rind[np.nonzero(extended_rind > len(sigma_kept)-1)] =\
            (len(sigma_kept)-1)*2 -\
            extended_rind[np.nonzero(extended_rind > len(sigma_kept)-1)]
        # remove duplicates (https://stackoverflow.com/a/36237337/9368839):
        _base = extended_sind.max()+1
        _combi = extended_rind + _base * extended_sind
        _vall, _indd = np.unique(_combi, return_index=True)
        _indd.sort()
        extended_sind = extended_sind[_indd]
        extended_rind = extended_rind[_indd]
        other_candidates = (extended_sind, extended_rind)
        mock_sp3[other_candidates] = median_spectra3[other_candidates]

        CR_cand_ind = np.unique(sind)
        #CR_cand_ind = np.arange(len(spectra_kept))
        _ss = np.stack((normalized_spectra[CR_cand_ind],
                        mock_sp3[CR_cand_ind]), axis=-1)
        check_CR_candidates = NavigationButtons(sigma_kept, _ss,
                                                autoscale_y=True,
                                                title=[
                                                    f"indice={i}" for i in CR_cand_ind],
                                                label=['normalized spectra',
                                                       'median correction'])
        if len(CR_cand_ind) > 10:
            plt.figure()
            sns.violinplot(y=rind)
            plt.title("Distribution of Cosmic Rays")
            plt.ylabel("CCD pixel struck")
    else:
        print("No Cosmic Rays found!")
    return mock_sp3


class AdjustCR_SearchSensitivity(object):
    """Visually set the sensitivity for the Cosmic Rays detection.

    The graph shows the number and the distribution of CR candidates along the
    Raman shifts' axis. You can manually adjust the sensitivity
    (left=more sensitive, right=less sensitive)

    The usage example is the following:
    ---------------------------------------
    first you show the graph and set for the appropriate sensitivity value:
    >>> my_class_instance = AdjustCR_SearchSensitivity(spectra, x_values=sigma)
    Once you're satisfied with the result, you should recover the following
    values:
    >>> CR_spectra_ind = my_class_instance.CR_spectra_ind
    >>> mask_CR_cand = my_class_instance.mask_CR_cand
    >>> mask_whole = my_class_instance.mask_whole

    The recovered values are:
    CR_spectra_ind: 1D ndarray of ints: The indices of the spectra containing
                                        the Cosmic Rays.
                                        It's length is the number of CRs found.
    mask_CR_cand: 2D ndarray of bools:  Boolean mask of the same shape as the
                                        spectra containing the CRs.
                                        shape = (len(CR_spectra_ind), len(x_values))
                                        Is True in the zone containing the CR.
    mask_whole: 2D ndarray of bools::   Boolean mask of the same shape as the
                                        input spectra. True where the CRs are.
    """

    def __init__(self, spectra, x_values=None, gradient_axis=-1):
        self.osa = gradient_axis
        self.spectra = spectra
        if x_values is None:
            self.x_values = np.arange(self.spectra.shape[-1])
        else:
            self.x_values = x_values
        assert len(x_values) == self.spectra.shape[-1], "wtf dude?"
        self.fig, self.ax = plt.subplots()
        # third gradient of the spectra (along the wavenumbers)
        self.nabla = np.gradient(np.gradient(np.gradient(self.spectra,
                                                         axis=self.osa),
                                             axis=self.osa),
                                 axis=self.osa)  # third gradient
        self.nabla_dev = np.std(self.nabla, axis=self.osa)
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes([0.15, 0.1, 0.7, 0.03],
                                         facecolor=self.axcolor)
        self.sframe = Slider(self.axframe, 'Sensitivity',
                             1, 22,
                             valinit=8, valfmt='%.1f', valstep=0.1)
        # calls the "update" function when changing the slider position
        self.sframe.on_changed(self.update)
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        self.CR_spectra_ind, self.mask_whole, self.mask_CR_cand = \
            self.calculate_mask(8)
        self.line, = self.ax.plot(
            self.x_values, np.sum(self.mask_whole, axis=-0))
        self.ax.set_title(f"Found {len(self.CR_spectra_ind)} cosmic rays")
        plt.show()

    def calculate_mask(self, CR_coeff):
        self.uslov = CR_coeff*self.nabla_dev[:, np.newaxis]
        # find the indices of the potential CR candidates:
        self.cand_spectra, self.cand_sigma =\
            np.nonzero(np.abs(self.nabla) > self.uslov)

        # indices of spectra containing the CR candidates:
        self.CR_spectra_ind = np.unique(self.cand_spectra)
        # we construct the mask with zeros everywhere except on the positions of CRs:
        self.mask_whole = np.zeros_like(self.spectra, dtype=bool)
        self.mask_whole[self.cand_spectra, self.cand_sigma] = True
        # we now dilate the mask:
        # the size of the window depends on resolution
        self.ws = int(self.spectra.shape[-1]/10)
        self.mask_CR_cand = scipy.ndimage.morphology.binary_dilation(
            self.mask_whole[self.CR_spectra_ind],
            structure=np.ones((1, self.ws)))
        self.mask_whole[self.CR_spectra_ind] = self.mask_CR_cand
        return self.CR_spectra_ind, self.mask_whole, self.mask_CR_cand

    def update(self, val):
        """Scroll through frames with a slider."""
        self.CR_coeff = self.sframe.val
        self.CR_spectra_ind, self.mask_whole, self.mask_CR_cand =\
            self.calculate_mask(self.CR_coeff)
        self.line.set_ydata(np.sum(self.mask_whole, axis=-0))
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(f"Found {len(self.CR_spectra_ind)} cosmic rays")
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use arrow keys left and right to scroll through frames one by one."""
        frame = self.sframe.val
        if event.key == 'left' and frame > 1:
            new_frame = frame - 0.1
        elif event.key == 'right' and frame < 22:
            new_frame = frame + 0.1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        self.CR_coeff = new_frame
        self.CR_spectra_ind, self.mask_whole, self.mask_CR_cand =\
            self.calculate_mask(self.CR_coeff)
        self.line.set_ydata(np.sum(self.mask_whole, axis=-0))
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_title(f"Found {len(self.CR_spectra_ind)} cosmic rays")
        self.fig.canvas.draw_idle()
