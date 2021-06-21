#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:36:18 2021

@author: dejan
"""
import numpy as np
import visualize as vis
from read_WDF import read_WDF

filename = "../Data/blah/blah/blahblah.wdf"
spectra, x_values, params, map_params, origins = read_WDF(filename, verbose=False)
nx, ny = [v for v in map_params["NbSteps"] if v > 1]
gle = vis.ShowSelected(spectra.reshape(ny,nx,-1), x_values);
