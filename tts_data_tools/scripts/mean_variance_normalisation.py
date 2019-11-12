"""Extracts mean variance statistics from a feature.

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
from tts_data_tools.wav_gen.utils import compute_deltas


def add_arguments(parser):
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--data_dir", action="store", dest="data_dir", type=str, required=True,
                        help="Directory containing the `--feat_name` directory.")
    parser.add_argument("--feat_name", action="store", dest="feat_name", type=str, required=True,
                        help="Name of the feature to calculate normalisation parameters for.")
    parser.add_argument("--normalisation_of_deltas", action="store_true", dest="normalisation_of_deltas", default=False,
                        help="Also calculate the MVN parameters for the delta and delta-delta features.")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--npy_file", dest="npy_file", action="store_true", default=True,
                        help="Whether the files being loaded are in .npy files (or .txt).")
    parser.add_argument("--no-npy_file", dest="npy_file", action="store_false", help=argparse.SUPPRESS)


def calculate_mvn_parameters(feature_list, deltas=False):
    for i, feature in enumerate(feature_list):
        # Initialise the numpy accumulation arrays on the first item in the feature_list iterator.
        if i == 0:
            feat_dim = feature.shape[1]

            sums = np.zeros(feat_dim, dtype=feature.dtype)
            sum_squares = np.zeros(feat_dim, dtype=feature.dtype)
            counts = 0

            if deltas:
                delta_sums = np.zeros(feat_dim * 3, dtype=feature.dtype)
                delta_sum_squares = np.zeros(feat_dim * 3, dtype=feature.dtype)

        # Accumulate the current features
        sums += np.sum(feature, axis=0)
        sum_squares += np.sum(feature ** 2, axis=0)
        counts += feature.shape[0]

        if deltas:
            delta_feature = compute_deltas(feature)
            delta_sums += np.sum(delta_feature, axis=0)
            delta_sum_squares += np.sum(delta_feature ** 2, axis=0)

    # Calculate the global parameters.
    mean = sums / counts
    variance = (sum_squares - (sums ** 2) / counts) / counts
    # Any variance values less than zero are due to numerical precision, ensure they are set to zero.
    variance[variance < 0.] = 0.
    std_dev = np.sqrt(variance)

    mvn_params = {
        'mean': mean.tolist(),
        'std_dev': std_dev.tolist()
    }

    if deltas:
        # Calculate the global parameters for delta and delta-delta features.
        delta_mean = delta_sums / counts
        delta_variance = (delta_sum_squares - (delta_sums ** 2) / counts) / counts
        # Any variance values less than zero are due to numerical precision, ensure they are set to zero.
        delta_variance[delta_variance < 0.] = 0.
        delta_std_dev = np.sqrt(delta_variance)

        delta_mvn_params = {
            'mean': delta_mean.tolist(),
            'std_dev': delta_std_dev.tolist()
        }

    else:
        delta_mvn_params = None

    return mvn_params, delta_mvn_params


def process(data_dir, feat_name, id_list=None, is_npy=True, deltas=False, out_dir=None):
    """Calculates the mean-variance normalisation statistics from a directory of features.

    Args:
        data_dir (str): Root directory containing folders of features.
        feat_name (str): Name of the feature to be normalised.
        id_list (str): List of file names to process.
        deltas (bool): Also calculate the MVN parameters for the delta and delta-delta features.
        is_npy (bool): If True uses `file_io.load_bin`, otherwise uses `file_io.load_txt` to load each file.
        out_dir (str): Location to save the normalisation parameters to.
    """
    feat_dir = os.path.join(data_dir, feat_name)
    file_ids = utils.get_file_ids(id_list=id_list)

    if is_npy:
        feature_list = file_io.load_dir(file_io.load_bin, feat_dir, file_ids, feat_ext='npy')
    else:
        feature_list = file_io.load_dir(file_io.load_txt, feat_dir, file_ids, feat_ext='txt')

    mvn_params, delta_mvn_params = calculate_mvn_parameters(feature_list, deltas)

    # Possibly save the parameters to json files.
    if out_dir is not None:
        mvn_file_path = os.path.join(out_dir, f'{feat_name}_mvn.json')
        file_io.save_json(mvn_params, mvn_file_path)

        if deltas:
            delta_mvn_file_path = os.path.join(out_dir, f'{feat_name}_deltas_mvn.json')
            file_io.save_json(delta_mvn_params, delta_mvn_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extracts mean-variance statistics for features in a given directory.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.data_dir, args.feat_name, id_list=args.id_list,
            is_npy=args.npy_file, deltas=args.normalisation_of_deltas, out_dir=args.out_dir)


if __name__ == "__main__":
    main()




