#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:36:18 2021

@author: dejan
"""
import numpy as np
import calculate as cc
import visualize as vis
import simulate as sim

# %% Create some data:
# (You should replace this whole part and load your own spectra.)

x_values = np.linspace(150, 1300, 1015) # Create 1015 equally spaced points

# initial peak parameters:
mpar = [[40, 220, 100], # h, x0, w, r
        [122, 440, 80],
        [164, 550, 160],
        [40, 480, 340],
        [123, 550],
        [435, 900, 1300]]

spectra = sim.create_multiple_spectra(x_values, mpar,
                                      noise=0.2, noise_bias='smiley')

# %% This cell illustrates the different measurements for the map scans
gle = vis.ShowSelected(spectra.reshape(100,100,-1), x_values);

# %% This cell illustrates the search for the baseline:
bl = vis.FindBaseline(spectra, x_values)
