"""Runs the feature extraction on the waveforms and binarises the label files.

Usage:
    python process.py \
        [--lab_dir DIR] [--state_level] \
        [--wav_dir DIR] \
        [--id_list FILE] \
        --out_dir DIR
"""

import argparse
import os

from tts_data_tools import lab_features
from tts_data_tools import utils

from .mean_variance_normalisation import calculate_mvn_parameters
from .save_features import save_durations


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--calculate_normalisation", action="store_true", dest="calculate_normalisation", default=False,
                        help="Whether to automatically calculate MVN parameters after extracting durations.")
    lab_features.add_arguments(parser)


def extract_durations(file_ids, lab_dir, state_level):

    @utils.multithread(_lab_dir=lab_dir, _state_level=state_level)
    def extract(file_id, _lab_dir, _state_level):
        lab_path = os.path.join(_lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, _state_level)

        duration = label.phone_durations.reshape((-1, 1))
        return duration

    return extract(file_ids)


def process(lab_dir, id_list, out_dir, state_level, calculate_normalisation):
    """Processes label files in id_list, saves the durations to file.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
        calculate_normalisation (bool): Whether to calculate min-max normalisation parameters for the counter features.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)

    durations = extract_durations(file_ids, lab_dir, state_level)

    save_durations(file_ids, durations, out_dir)

    if calculate_normalisation:
        calculate_mvn_parameters(durations, out_dir, feat_name='dur', deltas=False)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract duration information from forced alignment label files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.id_list, args.out_dir,
            args.state_level, args.calculate_normalisation)


if __name__ == "__main__":
    main()

