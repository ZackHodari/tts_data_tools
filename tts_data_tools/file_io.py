"""Functional interface for loading/saving files with different formats.

Usage:
    python file_io.py \
        --in_file FILE [--in_file_encoding ENUM] \
        --out_file FILE [--out_file_encoding ENUM]
"""

import json
import os
from scipy.io import wavfile

import numpy as np

from tts_data_tools import utils


def load_dir(load_fn, path, file_ids, feat_ext=None):
    for file_id in file_ids:

        if feat_ext is not None:
            file_id = '{}.{}'.format(file_id, feat_ext)
        file_path = os.path.join(path, file_id)

        datum = load_fn(file_path)
        yield datum


def save_dir(save_fn, path, data, file_ids, feat_ext=None):
    utils.make_dirs(path, file_ids)

    for datum, file_id in zip(data, file_ids):

        if feat_ext is not None:
            file_id = '{}.{}'.format(file_id, feat_ext)
        file_path = os.path.join(path, file_id)

        save_fn(datum, file_path)


def sanitise_array(data):
    """Sanitises data to a numpy matrix of fixed shape, and ensures it has at most 2 axes.

    Args:
        data (list<_> or np.ndarray): Matrix/Vector/Scalar data in python lists (or as a numpy array).

    Returns:
        (np.ndarray): Sanitised numpy array with 2 axes."""
    array = np.array(data)

    if array.ndim == 0:
        array = array[np.newaxis, np.newaxis]
    elif array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim != 2:
        raise ValueError("Only 1/2 dimensional data can be saved to text files, data.shape = {}".format(array.shape))

    return array


#
# JSON.
#
def load_json(file_path):
    with open(file_path, 'r') as f:
        mean_variance_params = json.load(f)

    return mean_variance_params


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


#
# File id lists (strings in a .scp text file).
#
def load_lines(file_path):
    """Loads text data from a text file.

    Args:
        file_path (str): File to load the text from.

    Returns:
        (list<str>) Sequence of strings."""
    with open(file_path, 'r') as f:
        lines = list(filter(bool, map(str.strip, f.readlines())))

    return lines


def save_lines(lines, file_path):
    """Saves text in a text file.

    Args:
        lines (list<str>): Sequence of strings.
        file_path (str): File to save the text to."""
    lines = list(map(lambda x: '{}\n'.format(x), lines))

    with open(file_path, 'w') as f:
        f.writelines(lines)


#
# Waveforms.
#
def load_wav(file_path):
    """Loads wave data from wavfile.

    Args:
        file_path (str): File to load from.

    Returns:
        (np.ndarray) Waveform samples,
        (int) Sample rate of waveform."""
    sample_rate, data = wavfile.read(file_path)
    return data, sample_rate


def save_wav(data, file_path, sample_rate):
    """Saves wave data to wavfile.

    Args:
        data (np.ndarray): Waveform samples.
        file_path (str): File to save to."""
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    elif data.dtype not in [np.int32, np.int16, np.uint8]:
        raise ValueError("wavfile data must be np.float*, np.int32, np.int16, or np.uint8, got {}".format(data.dtype))

    wavfile.write(file_path, sample_rate, data)


#
# Numpy binary files.
#
def load_bin(file_path):
    """Loads data from a binary file using numpy.

    Args:
        file_path (str): File to load the data from.

    Returns:
        (np.ndarray) Sequence of frame-level vectors/floats/ints."""
    return np.load(file_path)


def save_bin(data, file_path):
    """Saves data as a binary file using numpy.

    Args:
        data (np.ndarray): Data to be saved.
        file_path (str): File to save the data to."""
    np.save(file_path, data)


#
# Numerical data saved using string delimiters in a .txt file.
#
def load_txt(file_path):
    """Loads data from a text file into a numpy array.

    Args:
        file_path (str): File to load the data from.

    Returns:
        (np.ndarray) Sequence of frame-level vectors/floats/ints."""
    lines = load_lines(file_path)

    if 'E' in lines[0]:
        dtype = np.float32
    else:
        dtype = np.int32

    data = list(map(str.split, lines))
    array = np.array(data, dtype=dtype)
    return array


def save_txt(data, file_path):
    """Saves data as a text file.

    If the data is floating point it is encoded into a string using scientific notation and 12 decimal places.

    Args:
        data (np.ndarray): Sequence of frame-level vectors/floats/ints.
        file_path (str): File to save the data to."""
    array = sanitise_array(data)

    # If the data is floating then format the values in scientific notation.
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
        formatter = lambda x: '{:.12E}'.format(x)
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int32)
        formatter = lambda x: str(x)
    else:
        raise TypeError("Type of the data could not be serialised - {}".format(array.dtype))

    lines = [' '.join(formatter(val) for val in row) + '\n' for row in array]
    with open(file_path, 'w') as f:
        f.writelines(lines)

