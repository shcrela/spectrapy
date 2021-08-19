# -*- coding: latin-1 -*-
from __future__ import print_function
import numpy as np
import os
import time
import pandas as pd
import constants_WDF_class as const
import visualize as vis

def convert_time(t):
    """Convert the Windows 64bit timestamp to human readable format.

    Input:
    -------
        t: timestamp in W64 format (default for .wdf files)
    Output:
    -------
        string formatted to suit local settings

    Example:
    -------
        >>> time_of_spectrum_recording =
          [convert_time(x) for x in origins.iloc[:,4]]

        should give you the list with the times on which
        each specific spectrum was recorded
    """
    return time.strftime('%c', time.gmtime((t/1e7-11644473600)))


def reorder(ar, nx, ny, method):
    if ar.ndim == 1:
        arr = ar.reshape(ny, nx)
    elif ar.ndim == 2:
        arr = ar.reshape(ny, nx, -1)
    else:
        print("WTF?!")
    if method == "InvertedRows":
        reordered = np.array([arr[i][::-1] if i&1 else arr[i] for i in range(ny)])
    elif method in ["Alternating", "StreamLine"]:
        reordered = np.rot90(arr, axes=(0, 1))
    else:
        reordered = arr
    return reordered.reshape(ar.shape)

def _read(f, dtype=np.uint32, count=1):
    """Reads bytes from binary file,
    with the most common values given as default.
    Returns the value itself if one value, or list if count > 1
    Note that you should do ".decode()"
    on strings to avoid getting strings like "b'string'"
    For further information, refer to numpy.fromfile() function
    """
    if count == 1:
        return np.fromfile(f, dtype=dtype, count=count)[0]
    else:
        return np.fromfile(f, dtype=dtype, count=count)[0:count]


class WDF(object):
    """
    Read data from the binary .wdf file.

    Parameters:
    -----------
    file: string
        full (absolute or relative) path to the .wdf file
    verbose: bool
        Weather you want to print the informations about the file.
    Attributes:
    -----------
    spectra: numpy array
        that's why we're here :)
    x_values: numpy array
        the x-axis of your spectra
    origins: pandas dataframe
        contains information about each individual point of measurement
    params: dict
        contains general informations about the measurement
    map_params: dict : (returned if the measurement is of type map)
        dictionary containing informations about the map
    n_x, n_y, n_z : ints
        number of steps in each direction (same as in map_params["NbSteps"])
    ncollected, nspectra: ints
        number of spectra collected, number of spectra expected
        same as: params["Count"], params["Capacity"]
    npoints: int
        number of points in each spectrum
        same as params["PointsPerSpectrum"]
        should be equal to len(x_values) = spectra.shape[-1]
    filename: string
        the name of the file (without the path)
    folder: string
        the folder containing the file
    b_off: list of ints
        offsets in bytes for each of the blocks found in the file
    block_names: list of strings
        names of each block found in the file
    block_sizes: list of ints
        sizes of each block in bytes




    """
    def __init__(self, file, verbose=False):

        self.folder, self.filename = os.path.split(file)
        self.verbose = verbose
        try:
            f = open(file, "rb")
            if self.verbose:
                print(f'Reading the file: \"{self.filename}\"\n')
        except IOError:
            raise IOError(f"File {file} does not exist!")
        self.filesize = os.path.getsize(file)
        self.block_names = []
        self.block_sizes = []
        self.b_off = []
        self.params = {}
        self.map_params = {}

        # Reading all of the block names, offsets and sizes
        offset = 0
        while offset < self.filesize - 1:
            f.seek(offset)
            self.b_off.append(offset)
            block_header = np.fromfile(f, dtype=const.HEADER_DT, count=1)
            offset += block_header['block_size'][0]
            self.block_names.append(block_header['block_name'][0].decode())
            self.block_sizes.append(block_header['block_size'][0])

        name = 'WDF1'
        gen = [i for i, x in enumerate(self.block_names) if x == name]
        for i in gen:
            self.print_block_header(name, i)
            f.seek(self.b_off[i]+16)
    #        TEST_WDF_FLAG = _read(f,np.uint64)
            self.params['WdfFlag'] = const.WDF_FLAGS[_read(f, np.uint64)]
            f.seek(60)
            self.params['PointsPerSpectrum'] = self.npoints = _read(f)
            # Number of spectra expected (nspectra):
            self.params['Capacity'] = self.nspectra = _read(f, np.uint64)
            # Number of spectra written into the file (ncollected):
            self.params['Count'] = self.ncollected = _read(f, np.uint64)
            # Number of accumulations per spectrum:
            self.params['AccumulationCount'] = _read(f)
            # Number of elements in the y-list (>1 for image):
            self.params['YlistLength'] = _read(f)
            self.params['XlistLength'] = _read(f)  # number of elements in the x-list
            self.params['DataOriginCount'] = _read(f)  # number of data origin lists
            self.params['ApplicationName'] = _read(f, '|S24').decode()
            version = _read(f, np.uint16, count=4)
            self.params['ApplicationVersion'] = '.'.join(
                [str(x) for x in version[0:-1]]) +\
                ' build ' + str(version[-1])
            self.params['ScanType'] = const.SCAN_TYPES[_read(f)]
            self.params['MeasurementType'] = const.MEASUREMENT_TYPES[_read(f)]
            self.params['StartTime'] = convert_time(_read(f, np.uint64))
            self.params['EndTime'] = convert_time(_read(f, np.uint64))
            self.params['SpectralUnits'] = const.DATA_UNITS[_read(f)]
            self.params['LaserWaveLength'] = np.round(10e6/_read(f, '<f'), 2)
            f.seek(240)
            self.params['Title'] = _read(f, '|S160').decode()
        if self.verbose:
            for key, val in self.params.items():
                print(f'{key:-<40s} : \t{val}')
            if self.nspectra != self.ncollected:
                print(f'\nATTENTION:\nNot all spectra were recorded\n'
                      f'Expected nspectra={self.nspectra},'
                      f'while ncollected={self.ncollected}'
                      f'\nThe {self.nspectra-self.ncollected} missing values'
                      f'will be shown as blanks\n')

        name = 'WMAP'
        gen = [i for i, x in enumerate(self.block_names) if x == name]
        for i in gen:
            self.print_block_header(name, i)
            f.seek(self.b_off[i] + 16)
            m_flag = _read(f)
            self.map_params['MapAreaType'] = const.MAP_TYPES[m_flag]  # _read(f)]
            _read(f)
            self.map_params['InitialCoordinates'] = np.round(_read(f, '<f', count=3), 2)
            self.map_params['StepSizes'] = np.round(_read(f, '<f', count=3), 2)
            self.map_params['NbSteps'] = self.n_x, self.n_y, self.n_z \
                                       = _read(f, np.uint32, count=3)
            self.map_params['LineFocusSize'] = _read(f)
        if self.verbose:
            for key, val in self.map_params.items():
                print(f'{key:-<40s} : \t{val}')

        name = 'DATA'
        gen = [i for i, x in enumerate(self.block_names) if x == name]
        for i in gen:
            data_points_count = self.npoints * self.ncollected
            self.print_block_header(name, i)
            f.seek(self.b_off[i] + 16)
            self.spectra = _read(f, '<f', count=data_points_count)\
                .reshape(self.ncollected, self.npoints)
            if verbose:
                print(f'{"The number of spectra":-<40s} : \t{self.spectra.shape[0]}')
                print(f'{"The number of points in each spectra":-<40s} : \t'
                      f'{self.spectra.shape[1]}')
            if self.params['MeasurementType'] == 'Map':
                if self.map_params['MapAreaType'] == 'InvertedRows':
                    self.spectra = [self.spectra[((xx//self.n_x)+1)*self.n_x-(xx % self.n_x)-1]
                               if (xx//self.n_x) % 2 == 1
                               else self.spectra[xx]
                               for xx in range(self.nspectra)]
                    self.spectra = np.asarray(self.spectra)
                    if verbose:
                        print('*It seems your file was recorded using the'
                              '"Inverted Rows" scan type'
                              '(sometimes also reffered to as "Snake").\n '
                              'Note that the spectra will be rearanged'
                              'so it could be read\n'
                              'the same way as other scan types'
                              '(from left to right, and from top to bottom)')
                if self.map_params['MapAreaType'] in ['Alternating', 'StreamLine']:
                    self.spectra = self.spectra.reshape(self.n_x, self.n_y, -1)
                    self.spectra = np.rot90(self.spectra, axes=(0, 1)).reshape(self.n_x*self.n_y, -1)

        name = 'XLST'
        gen = [i for i, x in enumerate(self.block_names) if x == name]
        for i in gen:
            self.print_block_header(name, i)
            f.seek(self.b_off[i] + 16)
            self.params['XlistDataType'] = const.DATA_TYPES[_read(f)]
            self.params['XlistDataUnits'] = const.DATA_UNITS[_read(f)]
            self.x_values = _read(f, '<f', count=self.npoints)
        if self.verbose:
            print(f"{'The shape of the x_values is':-<40s} : \t{self.x_values.shape}")
            print(f"*These are the \"{self.params['XlistDataType']}"
                  f"\" recordings in \"{self.params['XlistDataUnits']}\" units")

    # The next block is where the image is stored (if recorded)
    # When y_values_count > 1, there should be an image.
        name = 'YLST'
        gen = [i for i, x in enumerate(self.block_names) if x == name]
        for i in gen:
            self.print_block_header(name, i)
            f.seek(self.b_off[i] + 16)
            self.params['YlistDataType'] = const.DATA_TYPES[_read(f)]
            self.params['YlistDataUnits'] = const.DATA_UNITS[_read(f)]
            self.y_values_count = int((self.block_sizes[i]-24)/4)
            # if y_values_count > 1, we can say that this is the number of pixels
            # in the recorded microscope image
            if self.y_values_count > 1:
                self.y_values = _read(f, '<f', count=self.y_values_count)
                if self.verbose:
                    print("There seem to be the image recorded as well")
                    print(f"{'Its size is':-<40s} : \t{self.y_values.shape}")
            else:
                if self.verbose:
                    print("*No image was found.")

        name = 'ORGN'
        origin_labels = []
        origin_set_dtypes = []
        origin_set_units = []
        origin_values = np.empty((self.params['DataOriginCount'],
                                  self.nspectra), dtype='<d')
        gen = [i for i, x in enumerate(self.block_names) if x == name]
        for i in gen:
            self.print_block_header(name, i)
            f.seek(self.b_off[i] + 16)
            nb_origin_sets = _read(f)
            # The above is the same as params['DataOriginCount']
            for set_n in range(nb_origin_sets):
                data_type_flag = _read(f).astype(np.uint16)
                # not sure why I had to add the astype part,
                # but if I just read it as uint32, I got rubbish sometimes
                origin_set_dtypes.append(const.DATA_TYPES[data_type_flag])
                origin_set_units.append(const.DATA_UNITS[_read(f)])
                origin_labels.append(_read(f, '|S16').decode())
                if data_type_flag == 11:
                    origin_values[set_n] = _read(f, np.uint64, count=self.nspectra)
                    # special case for reading timestamps
                else:
                    origin_values[set_n] = np.round(
                        _read(f, '<d', count=self.nspectra), 2)

                if self.params['MeasurementType'] == 'Map':
                    if self.map_params['MapAreaType'] == 'InvertedRows':
                        # To put the "Inverted Rows" into the
                        # "from left to right" order
                        origin_values[set_n] = [origin_values[set_n]
                                                [((xx//self.n_x)+1)*self.n_x-(xx % self.n_x)-1]
                                                if (xx//self.n_x) % 2 == 1
                                                else origin_values[set_n][xx]
                                                for xx in range(self.nspectra)]
                        origin_values[set_n] = np.asarray(origin_values[set_n])
                    if self.map_params['MapAreaType']  in ['Alternating', 'StreamLine']:
                        ovl = origin_values[set_n].reshape(self.n_x, self.n_y)
                        origin_values[set_n] = np.rot90(ovl, axes=(0, 1)).ravel()
        if self.verbose:
            print('\n\n\n')
        self.origins = pd.DataFrame(origin_values.T,
                               columns=[f"{x} ({d})" for (x, d) in \
                                        zip(origin_labels, origin_set_units)])


    def print_block_header(self, name, i):
        if self.verbose:
            print(f"\n{' Block : '+ name + ' ':=^80s}\n"
                  f"size: {self.block_sizes[i]}, offset: {self.b_off[i]}")
