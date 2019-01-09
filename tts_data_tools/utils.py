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
        pool = ThreadPool()
        with tqdm_redirect_std() as orig_stdout:
            for _ in tqdm(pool.imap_unordered(func, args_list), total=len(args_list),
                          file=orig_stdout, dynamic_ncols=True):
                pass
        pool.close()
        pool.join()

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
        with tqdm_redirect_std() as orig_stdout:
            for args in tqdm(args_list, file=orig_stdout, dynamic_ncols=True):
                func(args)

    return wrapper


def get_file_ids(file_dir, id_list=None):
    """Determines the basenames of all files to be processed, using id_list of `os.listdir`.

    Args:
        file_dir (str): Directory where the basenames would exist.
        id_list (str): File containing a list of basenames, if not given `os.listdir(dir)` is used instead.

    Returns:
        file_ids (list<str>): Basenames of files in dir or id_list"""
    if id_list is None:
        # Ignore hidden files starting with a period, and remove file extensions.
        file_ids = filter(lambda f: not f.startswith('.'), os.listdir(file_dir))
        file_ids = list(map(lambda x: os.path.splitext(x)[0], file_ids))
    else:
        file_ids = load_lines(id_list)

    return file_ids

