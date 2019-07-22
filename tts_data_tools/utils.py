"""Utility functions for various tools."""

import os

import numpy as np

from tts_data_tools.file_io import load_lines


def listify(object_or_list):
    r"""Converts input to an iterable if it is not already one."""
    if not isinstance(object_or_list, (list, tuple)):
        object_or_list = [object_or_list]
    return object_or_list


def get_file_ids(file_dir=None, id_list=None):
    """Determines basenames of files id_list or `os.listdir`, checks there are no missing files.

    Args:
        file_dir (str): Directory where the basenames would exist.
        id_list (str): File containing a list of basenames, if not given `os.listdir(dir)` is used instead.

    Returns:
        file_ids (list<str>): Basenames of files in dir or id_list"""
    if file_dir is not None:
        # Ignore hidden files starting with a period, and remove file extensions.
        _file_ids = filter(lambda f: not f.startswith('.'), os.listdir(file_dir))
        _file_ids = list(map(lambda x: os.path.splitext(x)[0], _file_ids))

    if id_list is None:
        file_ids = _file_ids
    else:
        file_ids = load_lines(id_list)

        # Check that `file_ids` is a subset of `_file_ids`
        if (file_dir is not None) and (not set(file_ids).issubset(_file_ids)):
            raise ValueError("All basenames in id_list '{}' must be present in file_dir '{}'".format(id_list, file_dir))

    return file_ids


def string_to_ascii(strings, max_len=None):
    r"""Converts a list of strings to a NumPy array of integers (ASCII codes).

    Parameters
    ----------
    strings : str or list<str>
        Strings of ASCII characters.
    max_len : int, optional
        The maximum number of characters in any of the strings. If None, this will be calculated using `strings`.

    Returns
    -------
    encoded : np.ndarray, shape (num_lines, max_len), dtype (np.int8)
        ASCII codes of each character. Each row represents one item in the list `strings`.

    See Also
    --------
    ascii_to_string : Performs the opposite opteration to `string_to_ascii`.
    """
    strings = listify(strings)
    num_lines = len(strings)

    if max_len is None:
        max_num_chars = max(map(len, strings))
    else:
        max_num_chars = max_len

    # Padding is partially handled here as each string may have a different length.
    encoded = np.zeros((num_lines, max_num_chars), dtype=np.int8)

    # Convert the strings into ASCII integers.
    for i, line in enumerate(strings):
        ascii = list(map(ord, line))
        encoded[i, :len(line)] = ascii

    return encoded


def ascii_to_string(ascii):
    r"""Converts an array of ASCII codes to a list of strings (each string is a row from the array).

    If the input `ascii` is a NumPy array it will likely contain padding. Padding is encoded using 0, which is the ASCII
    code for the null character \x00. This character is stripped from the output strings.

    Parameters
    ----------
    ascii : array_like, shape (num_lines, max_len)
        ASCII codes of each character. Each row represents one item in the list `strings`.

    Returns
    -------
    strings : str or list<str>
        Strings of ASCII characters.

    See Also
    --------
    string_to_ascii : Performs the opposite opteration to `ascii_to_string`.
    """
    # Convert the ASCII codes into python strings.
    chars_list_with_padding = [map(chr, codes) for codes in ascii]

    # Remove the padding, which is stored as a zero, i.e. the null character \x00.
    chars_list = [filter(lambda s: s != chr(0), chars) for chars in chars_list_with_padding]

    # Return a list of strings.
    return [''.join(chars) for chars in chars_list]

