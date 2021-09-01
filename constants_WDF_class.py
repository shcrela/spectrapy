#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:43:34 2021

@author: dejan
"""
import numpy as np

BLOCK_NAMES = ['WDF1',
               'DATA',
               'YLST'
              ]

DATA_TYPES = ['Arbitrary',
              'Spectral',
              'Intensity',
              'SpatialX',
              'SpatialY',
              'SpatialZ',
              'SpatialR',
              'SpatialTheta',
              'SpatialPhi',
              'Temperature',
              'Pressure',
              'Time',
              'Derived',
              'Polarization',
              'FocusTrack',
              'RampRate',
              'Checksum',
              'Flags',
              'ElapsedTime',
              'Frequency',
              'MpWellSpatialX',
              'MpWellSpatialY',
              'MpLocationIndex',
              'MpWellReference',
              'PAFZActual',
              'PAFZError',
              'PAFSignalUsed',
              'ExposureTime',
              'EndMarker']

DATA_UNITS = ['Arbitrary',
              'RamanShift',
              'Wavenumber',
              'Nanometre',
              'ElectronVolt',
              'Micron',
              'Counts',
              'Electrons',
              'Millimetres',
              'Metres',
              'Kelvin',
              'Pascal',
              'Seconds',
              'Milliseconds',
              'Hours',
              'Days',
              'Pixels',
              'Intensity',
              'RelativeIntensity',
              'Degrees',
              'Radians',
              'Celcius',
              'Farenheit',
              'KelvinPerMinute',
              'FileTime',
              'Microseconds',
              'EndMarker']

SCAN_TYPES = ['Unspecified',
              'Static',
              'Continuous',
              'StepRepeat',
              'FilterScan',
              'FilterImage',
              'StreamLine',
              'StreamLineHR',
              'Point',
              'MultitrackDiscrete',
              'LineFocusMapping']

MAP_TYPES = {0: 'RandomPoints',
             1: 'ColumnMajor',
             2: 'Alternating2',
             3: 'LineFocusMapping',
             4: 'InvertedRows',
             5: 'InvertedColumns',
             6: 'SurfaceProfile',
             7: 'XyLine',
             66: 'StreamLine',
             68: 'InvertedRows',
             128: 'Slice'}
# Remember to check this 68

MEASUREMENT_TYPES = ['Unspecified',
                     'Single',
                     'Series',
                     'Map']

WDF_FLAGS = {0: 'WdfXYXY',
             1: 'WdfChecksum',
             2: 'WdfCosmicRayRemoval',
             3: 'WdfMultitrack',
             4: 'WdfSaturation',
             5: 'WdfFileBackup',
             6: 'WdfTemporary',
             7: 'WdfSlice',
             8: 'WdfPQ',
             16: 'UnknownFlag (check in WiRE?)'}


EXIF_TAGS = {
             # Renishaw's particular tags:
             65184: "FocalPlaneXYOrigins", # tuple of floats
             65185: "FieldOfViewXY",       # tuple of floats
             65186: "px/µ ?",           # float (1.0, 5.0?)
             # Standard Exif tags:
             34665: "ExifOffset",               # Normally, (114?)
               270: "ImageDescription",         # Normally, "white-light image"
               271: "Make",                     # Normally, "Renishaw"
             41488: "FocalPlaneResolutionUnit", # (`5` corresponds to microns)
             41486: "FocalPlaneXResolution",    # (27120.6?)
             41487: "FocalPlaneYResolution"}    # (21632.1?)


HEADER_DT = np.dtype([('block_name', '|S4'),
                      ('block_id', np.int32),
                      ('block_size', np.int64)])
