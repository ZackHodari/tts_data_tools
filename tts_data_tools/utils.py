"""Utility functions for various tools."""

from contextlib import contextmanager
from functools import wraps
from multiprocessing.pool import ThreadPool
import os
import sys
from tqdm import tqdm

from .file_io import load_lines


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


@contextmanager
def tqdm_redirect_std():
    """Context manager that redirects print statements to DummyTqdmFile instances, ensuring progress bar is visible."""
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]

    # Relay exceptions
    except Exception as exc:
        raise exc

    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def multithread(func):
    """Uses Python multithreading to perform func over arg_list in parallel.

    Args:
        func (callable): Python function that will be parallelised.

    Callable Args:
        args_list (list<args>): A list where each item are valid argument(s) for func, e.g. args_list can be file_ids.
    """
    @wraps(func)
    def wrapper(args_list):
        results = []
        pool = ThreadPool()
        with tqdm_redirect_std() as orig_stdout:
            for result in tqdm(pool.imap_unordered(func, args_list), total=len(args_list),
                               file=orig_stdout, dynamic_ncols=True):
                results.append(result)
        pool.close()
        pool.join()

        return results

    return wrapper


def singlethread(func):
    """Calls func for all items in args_list, but not in parallel.

    This function exists multithread decorator can be replaced without changing any other code.

    Args:
        func (callable): Python function that will be parallelised.

    Callable Args:
        args_list (list<args>): A list where each item are valid argument(s) for func, e.g. args_list can be file_ids.
    """
    @wraps(func)
    def wrapper(args_list):
        results = []
        with tqdm_redirect_std() as orig_stdout:
            for args in tqdm(args_list, file=orig_stdout, dynamic_ncols=True):
                result = func(args)
                results.append(result)

        return results

    return wrapper


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

