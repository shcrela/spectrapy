#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:34:02 2021

@author: dejan
"""
import numpy as np
from scipy import linalg, spatial
from sklearn import metrics, decomposition
import matplotlib.pyplot as plt
from read_WDF import read_WDF
import visualize as vis
import CR_search as cr
import preprocessing as pp
import calculate as cc

folder = "../../Raman/Data/Chloe/prob/"
file = "sableindien-x5-532streamline-map1.wdf"

# folder = "../../Raman/Data/Giuseppe/"
# file = "TFCD_ITOcell_532nm_p100_1s_carto_z20.wdf"
filename = folder + file

rawspectra, x_values, params, map_params, origins = read_WDF(filename)

# Order:
rawspectra, x_values = pp.order(rawspectra, x_values)

vis.ShowSpectra(rawspectra, x_values)

#%% Find saturated spectra
saturated_indices = pp.find_saturated(rawspectra)

# Replace the zeros with max values
for i in saturated_indices:
    rawspectra[i][rawspectra[i]==0] = np.max(rawspectra[i])

zero_indices = pp.find_zeros(rawspectra)
# Combine the saturated and zeros
zero_saturated_idx = np.union1d(saturated_indices, zero_indices)

if len(zero_saturated_idx) > 0:
    title = ["Spectra N° "+str(n) for n in zero_saturated_idx]
    saturated = vis.ShowSpectra(rawspectra[zero_saturated_idx], x_values, title=title)
    saturated.fig.suptitle("Saturated and zero spectra")
    # Replace all of the saturated values by the median spectra
    rawspectra[zero_saturated_idx] = np.median(rawspectra, axis=0)
# Normalize
norma = linalg.norm(rawspectra, axis=-1, keepdims=True)
norma[norma==0] = 1

n_spectra = rawspectra/norma
n_pectra = pp.scale(n_spectra)
#%%
# BAseline:
b_line = cc.baseline_als(n_spectra)#, lam=1e7, p=0.5)
spectra = n_spectra - b_line
vis.ShowSpectra(spectra, x_values)
#%%
# Sparse PCA
pca = decomposition.SparsePCA(n_components=4)
n_spectra = pp.scale(rawspectra)
reduced = pca.fit(n_spectra)

#%%
# calculate the distances between all pairs of spectra:
distances = spatial.distance.pdist(spectra, 'hamming')
dd = spatial.distance.squareform(distances)

dd = metrics.pairwise.nan_euclidean_distances(spectra, missing_values=0)
def closest(indice):
    global spectra, distances, dd, zero_saturated_idx

    a = dd[indice]
    valid_idx = np.setdiff1d(np.where(a > 0)[0], zero_saturated_idx)
    closest_idx = valid_idx[a[valid_idx].argmin()]
    plt.figure()
    plt.plot(spectra[indice], alpha=0.5, label=indice)
    plt.plot(spectra[closest_idx], alpha=0.5, label=closest_idx)
    plt.legend()


closest(834)
#%%
crs = cr.AdjustCR_SearchSensitivity(spectra, x_values)
ccc = cr.remove_CRs(mock_sp3, sigma_kept, initialization)
#cr.remove_cosmic_rays(spectra, x_values)
#%%
# We identify where are the sharp peaks:
surplus = np.diff(spectra, n=3, axis=-1, append=spectra[:,-3:])
condition = surplus / (spectra+1)
kandidati = np.where(surplus  > 5000)
# mean_values = np.mean(spectra, axis=-1, keepdims=True)
# kandidati = np.where(surplus / mean_values > 1)


# Now, we need to check if among `kandidati` we observe some repeating values,
# in which case they are more likely to represent cristalisations then CRs
cumuls = np.histogram(kandidati[1], bins=x_values[::10])
kristali = np.where(cumuls[0] > np.std(cumuls[0]))[0]

# Eliminate the repeating values from `kandidati`
left_bin_border = cumuls[1][kristali]
right_bin_border = cumuls[1][kristali+1]
to_keep = np.ones_like(kandidati[0], dtype=bool)
for i, (l, r) in enumerate(zip(left_bin_border, right_bin_border)):
    to_keep[np.where((l < kandidati[1]) & (kandidati[1] < r))[0]] = 0

CRs = (kandidati[0][to_keep], x_values[kandidati[1][to_keep]])
