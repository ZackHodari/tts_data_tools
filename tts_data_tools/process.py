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

from . import file_io
from . import lab_features
from . import proto_ops
from . import utils
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


def process_files(lab_dir, wav_dir, id_list, out_dir, state_level, question_file, subphone_feat_type):
    """Processes label and wave files in id_list, saves the numerical labels and vocoder features to a protobuffer.

    Args:
        lab_dir (str): Directory containing the label files.
        wav_dir (str): Directory containing the wave files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that label files are state level if True, otherwise they are frame level.
        question_file (str): Question set to be loaded. Can be one of the four provided question sets;
                questions-unilex_dnn_600.hed
                questions-radio_dnn_416.hed
                questions-mandarin.hed
                questions-japanese.hed
        subphone_feat_type (str): Subphone features to be extracted from the durations. If None, then no additional
            frame-level features are added.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)
    _file_ids = utils.get_file_ids(wav_dir, id_list)

    if len(file_ids) != len(_file_ids) or sorted(file_ids) != sorted(_file_ids):
        raise ValueError("Please provide id_list, or ensure that wav_dir and lab_dir contain the same files.")

    os.makedirs(out_dir, exist_ok=True)

    questions = lab_features.QuestionSet(question_file)
    suphone_features = lab_features.SubphoneFeatureSet(subphone_feat_type)

    @utils.multithread
    def save_lab_and_wav_to_proto(file_id):
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, state_level)
        numerical_labels = label.normalise(questions, suphone_features)

        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)
        f0, sp, ap = wav.extract_features()

        # Often there is a small difference in number of frames between labels and vocoder features
        n_frames = min(numerical_labels.shape[0], f0.shape[0])

        features = {
            'lab': numerical_labels[:n_frames],
            'dur': label.phone_durations.reshape((-1, 1)),
            'f0': f0[:n_frames],
            'sp': sp[:n_frames],
            'ap': ap[:n_frames]
        }
        context = {
            'name': file_id,
            'seq_len': np.array([n_frames])
        }
        proto = proto_ops.arrays_to_SequenceExample(features, context)

        feature_path = os.path.join(out_dir, '{}.proto'.format(file_id))
        file_io.save_proto(proto, feature_path)

    save_lab_and_wav_to_proto(file_ids)


def process_lab_files(lab_dir, id_list, out_dir, state_level, question_file, subphone_feat_type):
    """Processes label files in id_list, saves the numerical labels and durations.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
        question_file (str): Question set to be loaded. Can be one of the four provided question sets;
                questions-unilex_dnn_600.hed
                questions-radio_dnn_416.hed
                questions-mandarin.hed
                questions-japanese.hed
        subphone_feat_type (str): Subphone features to be extracted from the durations. If None, then no additional
            frame-level features are added.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)

    os.makedirs(os.path.join(out_dir, 'lab'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'dur'), exist_ok=True)

    questions = lab_features.QuestionSet(question_file)
    suphone_features = lab_features.SubphoneFeatureSet(subphone_feat_type)

    @utils.multithread
    def save_lab_and_dur_to_files(file_id):
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, state_level)

        duration_path = os.path.join(out_dir, 'dur', '{}.dur'.format(file_id))
        duration = label.phone_durations
        file_io.save_txt(duration, duration_path)

        numerical_label_path = os.path.join(out_dir, 'lab', '{}.lab'.format(file_id))
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
    file_ids = utils.get_file_ids(wav_dir, id_list)

    os.makedirs(os.path.join(out_dir, 'f0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mgc'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'bap'), exist_ok=True)

    @utils.multithread
    def save_wav_to_files(file_id):
        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)

        f0, mgc, bap = wav.extract_features()

        file_io.save_bin(f0, os.path.join(out_dir, 'f0', '{}.f0'.format(file_id)))
        file_io.save_bin(mgc, os.path.join(out_dir, 'mgc', '{}.mgc'.format(file_id)))
        file_io.save_bin(bap, os.path.join(out_dir, 'bap', '{}.bap'.format(file_id)))

    save_wav_to_files(file_ids)


def calclate_mvn_parameters_from_protos(protos_dir, id_list, mvn_file_path, mvn_keys=('f0', 'sp', 'ap')):
    """Calculates the mean-variance normalisation statistics from a directory of SequenceExample protos.

    Args:
        protos_dir (str): Directory containing the wave files.
        id_list (str): List of proto file names to process.
        mvn_file_path (str): File to save the mean-variance normalisation parameters to.
        mvn_keys (list<str>): Names of the features to calculate mean-variance normalisation parameters for.
    """
    file_ids = utils.get_file_ids(protos_dir, id_list)
    get_file_path = lambda file_id: '{}/{}.proto'.format(protos_dir, file_id)
    protos = (file_io.load_proto(get_file_path(file_id)) for file_id in file_ids)

    return calclate_mvn_parameters(protos, mvn_file_path, mvn_keys)


def calclate_mvn_parameters_from_tfrecord(tfrecord_path, mvn_file_path, mvn_keys=('f0', 'sp', 'ap')):
    """Calculates the mean-variance normalisation statistics from a directory of SequenceExample protos.

    Args:
        tfrecord_path (str): Directory containing the wave files.
        mvn_file_path (str): File to save the mean-variance normalisation parameters to.
        mvn_keys (list<str>): Names of the features to calculate mean-variance normalisation parameters for.
    """
    protos = file_io.load_TFRecord(tfrecord_path)

    return calclate_mvn_parameters(protos, mvn_file_path, mvn_keys)


def calclate_mvn_parameters(protos, mvn_file_path, mvn_keys=('f0', 'sp', 'ap')):
    """Calculates the mean-variance normalisation statistics from a list of SequenceExample protos.

    Args:
        mvn_file_path (str): File to save the mean-variance normalisation parameters to.
        mvn_keys (list<str>): Names of the features to calculate mean-variance normalisation parameters for.
    """
    features = {}
    for proto in protos:
        data, _ = proto_ops.SequenceExample_to_arrays(proto)
        for key, array in data.items():
            if key not in features:
                features[key] = array
            else:
                features[key] = np.concatenate((features[key], array), axis=0)

    mvn_params = {}
    for feature_name in mvn_keys:
        feature = features[feature_name]
        n_frames = feature.shape[0]

        mean = np.sum(feature, axis=0) / n_frames
        variance = np.sum((feature - mean) ** 2, axis=0) / n_frames

        mvn_params[feature_name] = {
            'mean': mean.tolist(),
            'variance': variance.tolist()
        }

    file_io.save_json(mvn_params, mvn_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract duration information from forced alignment label files.")
    add_arguments(parser)
    args = parser.parse_args()

    if args.lab_dir and args.wav_dir:
        process_files(args.lab_dir, args.wav_dir, args.id_list, args.out_dir, args.state_level,
                      args.question_file, args.subphone_feat_type)

    elif args.lab_dir:
        process_lab_files(args.lab_dir, args.id_list, args.out_dir, args.state_level,
                          args.question_file, args.subphone_feat_type)

    elif args.wav_dir:
        process_wav_files(args.wav_dir, args.id_list, args.out_dir)


if __name__ == "__main__":
    main()
