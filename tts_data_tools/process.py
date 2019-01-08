"""Runs the feature extraction on the waveforms and binarises the label files.

Usage:
    python process.py \
        [--lab_dir DIR] [--state_level] \
        [--wav_dir DIR] \
        [--id_list FILE] \
        --out_dir DIR
"""

import argparse
from functools import wraps
from multiprocessing.pool import ThreadPool
import os
from tqdm import tqdm

from . import file_io
from . import lab_features
from . import wav_features


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, default=None,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--wav_dir", action="store", dest="wav_dir", type=str, default=None,
                        help="Directory of the wave files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True, default=None,
                        help="Directory to save the output to.")
    file_io.add_arguments(parser)
    lab_features.add_arguments(parser)


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
        for _ in tqdm(pool.imap_unordered(func, args_list), total=len(args_list)):
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
        for args in tqdm(args_list):
            func(args)

    return wrapper


def get_file_ids(dir, id_list=None):
    """Determines the basenames of all files to be processed, using id_list of `os.listdir`.

    Args:
        dir (str): Directory where the basenames would exist.
        id_list (str): File containing a list of basenames, if not given `os.listdir(dir)` is used instead.

    Returns:
        file_ids (list<str>): Basenames of files in dir or id_list"""
    if id_list is None:
        file_ids = filter(lambda f: f.endswith('.lab'), os.listdir(dir))
        file_ids = map(lambda x: x[:-len('.lab')], file_ids)
    else:
        file_ids = lab_features.load_txt(id_list)

    return file_ids


def process_files(lab_dir, wav_dir, id_list, out_dir, state_level, questions, suphone_features):
    """Processes label and wave files in id_list, saves the numerical labels and vocoder features to a protobuffer.

    Args:
        lab_dir (str): Directory containing the label files.
        wav_dir (str): Directory containing the wave files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that label files are state level if True, otherwise they are frame level.
        questions (label_io.QuestionSet instance): Question set used to query the labels.
        suphone_features (label_io.SubphoneFeatureSet instance): Container that defines the subphone features to be
            extracted from the durations. If None, then no additional frame-level features are added.
        """
    file_ids = get_file_ids(lab_dir, id_list)
    _file_ids = get_file_ids(wav_dir, id_list)

    if file_ids != _file_ids:
        raise ValueError("Please provide id_list, or ensure that wav_dir and lab_dir contain the same files.")

    @multithread
    def save_lab_and_wav_to_proto(file_id):
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, state_level)
        numerical_labels = label.normalise(questions, suphone_features)

        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)
        f0, sp, ap = wav.extract_features()

        features = {
            'lab': numerical_labels,
            'duration': label.phone_durations,
            'f0': f0,
            'sp': sp,
            'ap': ap
        }

        feature_path = os.path.join(out_dir, '{}.proto'.format(file_id))
        file_io.save_proto(features, feature_path)

    save_lab_and_wav_to_proto(file_ids)


def process_lab_files(lab_dir, id_list, out_dir, state_level, questions, suphone_features):
    """Processes label files in id_list, saves the numerical labels and durations.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
        questions (`label_io.QuestionSet` instance): Question set used to query the labels.
        suphone_features (`label_io.SubphoneFeatureSet` instance): Container that defines the subphone features to be
            extracted from the durations. If None, then no additional frame-level features are added.
        """
    file_ids = get_file_ids(lab_dir, id_list)

    @multithread
    def save_lab_and_dur_to_files(file_id):
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, state_level)

        duration_path = os.path.join(out_dir, '{}.dur'.format(file_id))
        duration = label.phone_durations
        file_io.save_txt(duration, duration_path)

        numerical_label_path = os.path.join(out_dir, '{}.lab'.format(file_id))
        numerical_labels = label.normalise(questions, suphone_features)
        file_io.save_bin(numerical_labels, numerical_label_path)

    save_lab_and_dur_to_files(file_ids)


def process_wav_files(wav_dir, id_list, out_dir):
    """Processes wave files in id_list, saves the vocoder features to binary numpy files.

    Args:
        wav_dir (str): Directory containing the wave files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        """
    file_ids = get_file_ids(wav_dir, id_list)

    @multithread
    def save_wav_to_files(file_id):
        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)

        f0, mgc, bap = wav.extract_features()

        feature_path = os.path.join(out_dir, file_id)
        file_io.save_bin(f0, '{}.f0'.format(feature_path))
        file_io.save_bin(mgc, '{}.mgc'.format(feature_path))
        file_io.save_bin(bap, '{}.bap'.format(feature_path))

    save_wav_to_files(file_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to extract duration information from forced alignment label files.")
    add_arguments(parser)
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.question_file:
        questions = lab_features.QuestionSet(args.question_file)
    else:
        questions = None

    if args.subphone_feat_type:
        suphone_features = lab_features.SubphoneFeatureSet(args.subphone_feat_type)
    else:
        suphone_features = None

    if args.lab_dir and args.wav_dir:
        process_files(
            args.lab_dir, args.wav_dir, args.id_list, args.out_dir, args.state_level, questions, suphone_features)

    elif args.lab_dir:
        process_lab_files(args.lab_dir, args.id_list, args.out_dir, args.state_level, questions, suphone_features)

    elif args.wav_dir:
        process_wav_files(args.wav_dir, args.id_list, args.out_dir)

