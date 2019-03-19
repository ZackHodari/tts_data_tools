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

from .min_max_normalisation import calculate_minmax_parameters
from .save_features import save_counter_features


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--subphone_feat_type", action="store", dest="subphone_feat_type", type=str, required=True,
                        help="The type of subphone counter features.")
    parser.add_argument("--calculate_normalisation", action="store_true", dest="calculate_normalisation", default=False,
                        help="Whether to automatically calculate min-max parameters after extracting counter features.")
    lab_features.add_arguments(parser)


def extract_counter_features(file_ids, lab_dir, state_level, subphone_feat_type):
    subphone_feature_set = lab_features.SubphoneFeatureSet(subphone_feat_type)

    @utils.multithread(_lab_dir=lab_dir, _state_level=state_level, _subphone_feature_set=subphone_feature_set)
    def extract(file_id, _lab_dir, _state_level, _subphone_feature_set):
        lab_path = os.path.join(_lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, _state_level)

        counter_feature = label.extract_counter_features(_subphone_feature_set)
        return counter_feature

    return extract(file_ids)


def process(lab_dir, id_list, out_dir, state_level, subphone_feat_type, calculate_normalisation):
    """Processes label files in id_list, saves the numerical labels and durations.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
        subphone_feat_type (str): Subphone features to be extracted from the durations.
        calculate_normalisation (bool): Whether to calculate min-max normalisation parameters for the counter features.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)

    counter_features = extract_counter_features(file_ids, lab_dir, state_level, subphone_feat_type)

    save_counter_features(file_ids, counter_features, out_dir)

    if calculate_normalisation:
        calculate_minmax_parameters(counter_features, out_dir, feat_name='counters')


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract duration information from forced alignment label files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.id_list, args.out_dir,
            args.state_level, args.subphone_feat_type, args.calculate_normalisation)


if __name__ == "__main__":
    main()
