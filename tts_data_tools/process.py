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
    """Processes label and wave files in id_list, saves the numerical labels and vocoder features to .npy binary files.

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
        subphone_feat_type (str): Subphone features to be extracted from the durations. If None, will be ignored.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)
    _file_ids = utils.get_file_ids(wav_dir, id_list)

    if len(file_ids) != len(_file_ids) or sorted(file_ids) != sorted(_file_ids):
        raise ValueError("Please provide id_list, or ensure that wav_dir and lab_dir contain the same files.")

    os.makedirs(out_dir, exist_ok=True)

    os.makedirs(os.path.join(out_dir, 'lab'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'dur'), exist_ok=True)
    if subphone_feat_type is not None:
        os.makedirs(os.path.join(out_dir, 'counts'), exist_ok=True)

    os.makedirs(os.path.join(out_dir, 'f0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'vuv'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'sp'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'ap'), exist_ok=True)

    os.makedirs(os.path.join(out_dir, 'n_frames'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'n_phones'), exist_ok=True)

    questions = lab_features.QuestionSet(question_file)
    subphone_features = lab_features.SubphoneFeatureSet(subphone_feat_type)

    @utils.multithread
    def save_lab_and_wav_to_files(file_id):
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, state_level)

        if subphone_feat_type is None:
            numerical_labels = label.normalise(questions, upsample_to_frame_level=False)
        else:
            numerical_labels, counter_features = label.normalise(questions, subphone_features, False)

        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)
        f0, vuv, sp, ap = wav.extract_features()

        # Often there is a small difference in number of frames between labels and vocoder features.
        durations = label.phone_durations
        n_frames = sum(durations)

        if f0.shape[0] < n_frames:
            # Remove excess durations if there is a shape mismatch.
            while sum(durations) != f0.shape[0]:
                # If the excess frames is more than the number of phones, the while loop will make multiple passes.
                diff = sum(durations) - f0.shape[0]
                # Remove 1 frame from each phone's duration starting at the end of the sequence.
                durations[-diff:] -= 1

            n_frames = f0.shape[0]

        features = dict()

        features['lab'] = numerical_labels[:n_frames]
        features['dur'] = durations.reshape((-1, 1))
        if subphone_feat_type is not None:
            features['counters'] = counter_features

        features['f0'] = f0[:n_frames]
        features['vuv'] = vuv[:n_frames]
        features['sp'] = sp[:n_frames]
        features['ap'] = ap[:n_frames]

        features['n_frames'] = np.array([n_frames])
        features['n_phones'] = np.array([len(label.phones)])

        for name, value in features.items():
            path = os.path.join(out_dir, name, '{}.{}'.format(file_id, name))
            file_io.save_bin(value, path)

    save_lab_and_wav_to_files(file_ids)


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
    subphone_features = lab_features.SubphoneFeatureSet(subphone_feat_type)

    @utils.multithread
    def save_lab_and_dur_to_files(file_id):
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, state_level)

        duration_path = os.path.join(out_dir, 'dur', '{}.dur'.format(file_id))
        duration = label.phone_durations
        file_io.save_txt(duration, duration_path)

        numerical_label_path = os.path.join(out_dir, 'lab', '{}.lab'.format(file_id))
        numerical_labels = label.normalise(questions, subphone_features)
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
    os.makedirs(os.path.join(out_dir, 'vuv'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'sp'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'ap'), exist_ok=True)

    @utils.multithread
    def save_wav_to_files(file_id):
        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)

        f0, vuv, sp, ap = wav.extract_features()

        file_io.save_bin(f0, os.path.join(out_dir, 'f0', '{}.f0'.format(file_id)))
        file_io.save_bin(vuv, os.path.join(out_dir, 'vuv', '{}.vuv'.format(file_id)))
        file_io.save_bin(sp, os.path.join(out_dir, 'sp', '{}.sp'.format(file_id)))
        file_io.save_bin(ap, os.path.join(out_dir, 'ap', '{}.ap'.format(file_id)))

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
        protos (list<tf.SequenceExample> or generator): A list of protos. If the dataset is large then use a generator.
        mvn_file_path (str): File to save the mean-variance normalisation parameters to.
        mvn_keys (list<str>): Names of the features to calculate mean-variance normalisation parameters for.
    """
    sums = {}
    sum_squares = {}
    counts = {}
    # Iterate through protos, accumulating information; since concatenating all data in memory may not be possible.
    for proto in protos:
        data, context = proto_ops.SequenceExample_to_arrays(proto)

        for feature_name in mvn_keys:
            feature = data[feature_name]

            if feature_name not in sums:
                sums[feature_name] = np.zeros(feature.shape[1:], dtype=np.float32)
            if feature_name not in sum_squares:
                sum_squares[feature_name] = np.zeros(feature.shape[1:], dtype=np.float32)
            if feature_name not in counts:
                counts[feature_name] = np.array([0], dtype=np.int32)

            sums[feature_name] += np.sum(feature, axis=0)
            sum_squares[feature_name] += np.sum(feature ** 2, axis=0)
            counts[feature_name] += context['n_frames']

    mvn_params = {}
    for feature_name in mvn_keys:
        sum_x = sums[feature_name]
        sum_square_x = sum_squares[feature_name]
        count_x = counts[feature_name]

        mean = sum_x / count_x
        variance = (sum_square_x - (sum_x ** 2) / count_x) / count_x
        std_dev = np.sqrt(variance)

        mvn_params[feature_name] = {
            'mean': mean.tolist(),
            'std_dev': std_dev.tolist()
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
