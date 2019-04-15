"""Runs the feature extraction on the waveforms and binarises the label files.

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

from tts_data_tools import lab_features
from tts_data_tools import utils
from tts_data_tools import wav_features

from tts_data_tools.scripts.mean_variance_normalisation import calculate_mvn_parameters
from tts_data_tools.scripts.min_max_normalisation import calculate_minmax_parameters
from tts_data_tools.scripts.save_features import save_numerical_labels, save_counter_features, save_durations, \
    save_n_frames, save_n_phones, save_lf0, save_vuv, save_sp, save_ap


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--wav_dir", action="store", dest="wav_dir", type=str, required=True,
                        help="Directory of the wave files to be converted.")
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
                        help="Whether to automatically calculate MVN parameters after extracting F0.")
    parser.add_argument("--normalisation_of_deltas", action="store_true", dest="normalisation_of_deltas", default=False,
                        help="Also calculate the MVN parameters for the delta and delta delta features.")
    lab_features.add_arguments(parser)


def extract_from_dataset(file_ids, lab_dir, wav_dir, state_level, question_file, upsample, subphone_feat_type):
    question_set = lab_features.QuestionSet(question_file)
    subphone_feature_set = lab_features.SubphoneFeatureSet(subphone_feat_type)

    @utils.multithread(_lab_dir=lab_dir, _wav_dir=wav_dir, _state_level=state_level,
                       _question_set=question_set, _upsample=upsample,
                       _subphone_feature_set=subphone_feature_set)
    def extract(file_id, _lab_dir, _wav_dir, _state_level, _question_set, _upsample, _subphone_feature_set):
        lab_path = os.path.join(_lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, _state_level)

        wav_path = os.path.join(_wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)

        numerical_label = label.extract_numerical_labels(_question_set, upsample_to_frame_level=_upsample)
        counter_feature = label.extract_counter_features(_subphone_feature_set)
        duration = label.phone_durations.reshape((-1, 1))
        n_frame = np.sum(duration).item()
        n_phone = len(label.phones)

        f0, vuv, sp, ap = wav.extract_features()
        lf0 = np.log(f0)

        # Often the durations from forced alignment are a few frames longer than the vocoder features.
        diff = n_frame - f0.shape[0]
        if diff > len(duration):
            raise ValueError("Number of label frames and vocoder frames is too different for {name}\n"
                             "\tvocoder frames {voc}\n"
                             "\tlabel frames {lab}\n"
                             "\tnumber of phones {phones}"
                             .format(name=file_id, voc=f0.shape[0], lab=n_frame, phones=len(duration)))

        # Remove excess durations if there is a shape mismatch.
        if diff > 0:
            # Remove 1 frame from each phone's duration starting at the end of the sequence.
            duration[-diff:] -= 1
            n_frame = f0.shape[0]
            print("Cropped {} frames from durations and  for utterance {}".format(diff, file_id))

        assert n_frame == np.sum(duration).item()

        counter_feature = counter_feature[:n_frame]
        lf0 = lf0[:n_frame]
        vuv = vuv[:n_frame]
        sp = sp[:n_frame]
        ap = ap[:n_frame]

        return numerical_label, counter_feature, duration, n_frame, n_phone, lf0, vuv, sp, ap

    return zip(*extract(file_ids))


def process(lab_dir, wav_dir, id_list, out_dir,
            state_level, question_file, upsample, subphone_feat_type, calculate_normalisation, normalisation_of_deltas):
    """Processes wav files in id_list, saves the log-F0 and MVN parameters to files.

    Args:
        lab_dir (str): Directory containing the label files.
        wav_dir (str): Directory containing the wav files.
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
        calculate_normalisation (bool): Whether to automatically calculate MVN parameters after extracting F0.
        normalisation_of_deltas (bool): Also calculate the MVN parameters for the delta and delta delta features.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)
    _file_ids = utils.get_file_ids(wav_dir, id_list)

    if len(file_ids) != len(_file_ids) or sorted(file_ids) != sorted(_file_ids):
        raise ValueError("Please provide id_list, or ensure that wav_dir and lab_dir contain the same files.")

    numerical_labels, counter_features, durations, n_frames, n_phones, lf0_list, vuv_list, sp_list, ap_list = \
        extract_from_dataset(file_ids, lab_dir, wav_dir, state_level, question_file, upsample, subphone_feat_type)

    save_numerical_labels(file_ids, numerical_labels, out_dir)
    save_counter_features(file_ids, counter_features, out_dir)
    save_durations(file_ids, durations, out_dir)
    save_n_frames(file_ids, n_frames, out_dir)
    save_n_phones(file_ids, n_phones, out_dir)

    save_lf0(file_ids, lf0_list, out_dir)
    save_vuv(file_ids, vuv_list, out_dir)
    save_sp(file_ids, sp_list, out_dir)
    save_ap(file_ids, ap_list, out_dir)

    if calculate_normalisation:
        calculate_minmax_parameters(numerical_labels, out_dir, feat_name='lab')
        calculate_minmax_parameters(counter_features, out_dir, feat_name='counters')
        calculate_mvn_parameters(durations, out_dir, feat_name='dur', deltas=False)

        calculate_mvn_parameters(lf0_list, out_dir, feat_name='lf0', deltas=normalisation_of_deltas)
        calculate_mvn_parameters(sp_list, out_dir, feat_name='sp', deltas=normalisation_of_deltas)
        calculate_mvn_parameters(ap_list, out_dir, feat_name='ap', deltas=normalisation_of_deltas)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract linguistic and acoustic features ready for training an acoustic model.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.wav_dir, args.id_list, args.out_dir,
            args.state_level, args.question_file, args.upsample_to_frame_level, args.subphone_feat_type,
            args.calculate_normalisation, args.normalisation_of_deltas)


if __name__ == "__main__":
    main()

