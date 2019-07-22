"""Extracts min max statistics from a feature.

Usage:
    python process.py \
        [--lab_dir DIR] [--state_level] \
        [--wav_dir DIR] \
        [--id_list FILE] \
        --out_dir DIR
"""

import argparse
import numpy as np
import os

from tts_data_tools import file_io
from tts_data_tools import utils


def add_arguments(parser):
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to, this should contain the `--feat_name` directory.")
    parser.add_argument("--feat_name", action="store", dest="feat_name", type=str, required=True,
                        help="Name of the feature to calculate normalisation parameters for.")
    parser.add_argument("--npy_file", dest="npy_file", action="store_true", default=True,
                        help="Whether the files being loaded are in .npy files (or .txt).")
    parser.add_argument("--no-npy_file", dest="npy_file", action="store_false", help=argparse.SUPPRESS)


def calculate_minmax_parameters(feature_list, data_dir=None, feat_name=None):
    for i, feature in enumerate(feature_list):
        # Initialise the numpy accumulation arrays on the first item in the feature_list iterator.
        if i == 0:
            feat_dim = feature.shape[1]

            mmin = float('inf') * np.ones(feat_dim, dtype=feature.dtype)
            mmax = -float('inf') * np.ones(feat_dim, dtype=feature.dtype)

        # Accumulate the current features
        mmin = np.min((mmin, *feature), axis=0)
        mmax = np.max((mmax, *feature), axis=0)

    minmax_params = {
        'mmin': mmin.tolist(),
        'mmax': mmax.tolist()
    }

    if data_dir and feat_name:
        minmax_file_path = os.path.join(data_dir, '{}_minmax.json'.format(feat_name))
        file_io.save_json(minmax_params, minmax_file_path)

    return minmax_params


def process(data_dir, feat_name, id_list=None, is_npy=True, ext=None):
    """Calculates the min-max normalisation statistics from a directory of features.

    Args:
        data_dir (str): Root directory containing folders of features.
        feat_name (str): Name of the feature to be normalised.
        id_list (str): List of file names to process.
        is_npy (bool): If True uses `file_io.load_bin`, otherwise uses `file_io.load_txt` to load each file.
        ext (str): File extension of the saved features.
    """
    if ext is None:
        ext = feat_name

    feat_dir = os.path.join(data_dir, feat_name)
    file_ids = utils.get_file_ids(feat_dir, id_list)

    if is_npy:
        feature_list = file_io.load_dir(file_io.load_bin, feat_dir, file_ids, feat_ext=ext)
    else:
        feature_list = file_io.load_dir(file_io.load_txt, feat_dir, file_ids, feat_ext=ext)

    calculate_minmax_parameters(feature_list, data_dir, feat_name)


def main():
    parser = argparse.ArgumentParser(
        description="Extracts min-max statistics for features in a given directory.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.out_dir, args.feat_name, id_list=args.id_list, is_npy=args.npy_file)


if __name__ == "__main__":
    main()

