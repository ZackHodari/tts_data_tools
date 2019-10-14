# Copyright (c) 2007 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

r"""Read and write HTK feature files.

This module reads and writes the acoustic feature files used by HTK
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
# __version__ = "$Revision $"

from contextlib import contextmanager
from struct import unpack, pack

import numpy as np

# HTK file_io constants
# ---------------------

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0o000100  # has energy
_N = 0o000200  # absolute energy supressed
_D = 0o000400  # has delta coefficients
_A = 0o001000  # has acceleration (delta-delta) coefficients
_C = 0o002000  # is compressed
_Z = 0o004000  # has zero mean static coefficients
_K = 0o010000  # has CRC checksum
_O = 0o020000  # has 0th cepstral coefficient
_V = 0o040000  # has VQ data
_T = 0o100000  # has third differential coefficients

# ---------------------


@contextmanager
def open_htk(file_name, mode, veclen=13):
    r"""Open an HTK format feature file for reading or writing.

    The mode parameter is 'rb' (reading) or 'wb' (writing).
    """
    if mode in ('r', 'rb'):
        # `veclen` is ignored since it's in the file.
        file = HTKFeatRead(file_name)

    elif mode in ('w', 'wb'):
        file = HTKFeatWrite(file_name, veclen)

    else:
        raise Exception("mode must be 'r', 'rb', 'w', or 'wb'")

    try:
        yield file
    finally:
        file.close()


class HTKFeatRead(object):
    r"""Read HTK format feature files"""
    def __init__(self, file_name):
        self.file_name = file_name
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)

        self.file = open(self.file_name, 'rb')

        self.n_samples = None
        self.samp_period = None
        self.sampSize = None
        self.param_kind = None
        self.dtype = None
        self.veclen = None
        self.A = None
        self.B = None
        self.hdrlen = None

        self.read_header()

    def __iter__(self):
        self.file.seek(12, 0)
        return self

    def read_header(self):
        self.file.seek(0, 0)

        spam = self.file.read(12)

        self.n_samples, self.samp_period, self.sampSize, self.param_kind = unpack(">IIHH", spam)

        # Get coefficients for compressed data
        if self.param_kind & _C:
            self.dtype = 'h'
            self.veclen = self.sampSize / 2

            if self.param_kind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0

            else:
                self.A = np.fromfile(self.file, 'f', self.veclen)
                self.B = np.fromfile(self.file, 'f', self.veclen)

                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()

        else:
            self.dtype = 'f'
            self.veclen = self.sampSize / 4

        self.hdrlen = self.file.tell()
        self.veclen = int(self.veclen)

    def seek(self, idx):
        self.file.seek(self.hdrlen + idx * self.sampSize, 0)

    def __next__(self):
        vec = np.fromfile(self.file, self.dtype, self.veclen)

        if len(vec) == 0:
            raise StopIteration

        if self.swap:
            vec = vec.byteswap()

        # Uncompress data to floats if required
        if self.param_kind & _C:
            vec = (vec.astype('f') + self.B) / self.A

        return vec

    def read_vec(self):
        return next(self)

    def read_all(self):
        self.seek(0)

        data = np.fromfile(self.file, self.dtype)
        data = data.reshape((-1, self.veclen))

        if self.swap:
            data = data.byteswap()

        # Uncompress data to floats if required
        if self.param_kind & _C:
            data = (data.astype('f') + self.B) / self.A

        return data, self.n_samples


class HTKFeatWrite(object):
    r"""Write Sphinx-II format feature files"""
    def __init__(self, file_name, veclen=13, samp_period=100000, param_kind=(MFCC | _O)):
        self.file_name = file_name
        self.veclen = veclen
        self.samp_period = samp_period
        self.param_kind = param_kind

        self.samp_size = veclen * 4
        self.dtype = 'f'
        self.file_size = 0
        self.swap = unpack('=i', pack('>i', 42))[0] != 42

        self.file = open(self.file_name, 'wb')

        self.write_header()

    def __del__(self):
        self.close()

    def close(self):
        self.write_header()
        self.file.close()

    def write_header(self):
        self.file.seek(0,0)
        self.file.write(pack(">IIHH", self.file_size, self.samp_period, self.samp_size, self.param_kind))

    def write_vec(self, vec):
        if len(vec) != self.veclen:
            raise Exception(f'Vector length must be {self.veclen}, got {len(vec)}')

        if self.swap:
            np.array(vec, self.dtype).byteswap().tofile(self.file)

        else:
            np.array(vec, self.dtype).tofile(self.file)

        self.file_size += self.veclen

    def writ_eall(self, arr):
        for row in arr:
            self.write_vec(row)

