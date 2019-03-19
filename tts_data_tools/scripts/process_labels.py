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

from tts_data_tools import file_io
from tts_data_tools import lab_features
from tts_data_tools import utils

from .mean_variance_normalisation import calculate_mvn_parameters
from .min_max_normalisation import calculate_minmax_parameters
from .save_features import save_counter_features, save_durations, save_numerical_labels


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--question_file", action="store", dest="question_file", type=str, required=True,
                        help="File containing the '.hed' question set to query the labels with.")
    parser.add_argument("--upsample_to_frame_level", action="store_true", dest="upsample_to_frame_level", default=False,
                        help="Whether to upsample the numerical labels to frame-level.")
    parser.add_argument("--subphone_feat_type", action="store", dest="subphone_feat_type", type=str, default=None,
                        help="The type of subphone counter features.")
    parser.add_argument("--calculate_normalisation", action="store_true", dest="calculate_normalisation", default=False,
                        help="Whether to automatically calculate MVN parameters after extracting label features.")
    file_io.add_arguments(parser)
    lab_features.add_arguments(parser)


def extract_from_labels(file_ids, lab_dir, state_level, question_file, upsample, subphone_feat_type):
    question_set = lab_features.QuestionSet(question_file)
    subphone_feature_set = lab_features.SubphoneFeatureSet(subphone_feat_type)

    @utils.multithread(_lab_dir=lab_dir, _state_level=state_level,
                       _question_set=question_set, _upsample=upsample,
                       _subphone_feature_set=subphone_feature_set)
    def extract(file_id, _lab_dir, _state_level, _question_set, _upsample, _subphone_feature_set):
        lab_path = os.path.join(_lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, _state_level)

        numerical_label = label.extract_numerical_labels(_question_set, upsample_to_frame_level=_upsample)
        counter_feature = label.extract_counter_features(_subphone_feature_set)
        duration = label.phone_durations.reshape((-1, 1))

        return numerical_label, counter_feature, duration

    return zip(*extract(file_ids))


def process(lab_dir, id_list, out_dir, state_level,
            question_file, upsample, subphone_feat_type, calculate_normalisation):
    """Processes label files in id_list, saves the numerical labels and durations to file.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
        question_file (str): Question set to be loaded. Can be one of the four provided question sets;
                questions-unilex_dnn_600.hed
                questions-unilex_phones_69.hed
                questions-radio_dnn_416.hed
                questions-radio_phones_48.hed
                questions-mandarin.hed
                questions-japanese.hed
        upsample (bool): Whether to upsample phone-level numerical labels to frame-level.
        subphone_feat_type (str): Subphone features to be extracted from the durations.
        calculate_normalisation (bool): Calculate mean-variance and min-max normalisation for duration and labels.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)

    numerical_labels, counter_features, durations = extract_from_labels(
        file_ids, lab_dir, state_level, question_file, upsample, subphone_feat_type)

    save_numerical_labels(file_ids, numerical_labels, out_dir)
    save_counter_features(file_ids, counter_features, out_dir)
    save_durations(file_ids, durations, out_dir)

    if calculate_normalisation:
        calculate_minmax_parameters(numerical_labels, out_dir, feat_name='lab')
        calculate_minmax_parameters(counter_features, out_dir, feat_name='counters')
        calculate_mvn_parameters(durations, out_dir, feat_name='dur', deltas=False)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract duration information from forced alignment label files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.id_list, args.out_dir, args.state_level,
            args.question_file, args.upsample_to_frame_level, args.subphone_feat_type, args.calculate_normalisation)


if __name__ == "__main__":
    main()

