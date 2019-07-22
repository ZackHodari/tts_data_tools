"""Utility functions for various tools."""

import os

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

