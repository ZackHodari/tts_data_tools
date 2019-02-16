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
    parser.add_argument("--feat_name", action="store", dest="feat_name", type=str, default=None,
                        help="Name of the feature to calculate normalisation parameters for.")
    parser.add_argument("--auto_calc_mvn", action="store_true", dest="auto_calc_mvn", default=False,
                        help="Whether to automatically calculate MVN parameters after processing lab and wav files.")
    parser.add_argument("--file_is_txt", action="store_true", dest="file_is_txt", default=False,
                        help="Whether the files being loaded are in .txt files (not .npy files), used for MVN.")
    file_io.add_arguments(parser)
    lab_features.add_arguments(parser)


def process_files(lab_dir, wav_dir, id_list, out_dir, state_level, question_file, subphone_feat_type, calc_mvn=False):
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
        calc_mvn (bool): If True, mean-variance normalisation parameters are calculated for acoustic features.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)
    _file_ids = utils.get_file_ids(wav_dir, id_list)

    if len(file_ids) != len(_file_ids) or sorted(file_ids) != sorted(_file_ids):
        raise ValueError("Please provide id_list, or ensure that wav_dir and lab_dir contain the same files.")

    os.makedirs(out_dir, exist_ok=True)

    # Linguistic feature directories.
    os.makedirs(os.path.join(out_dir, 'lab'), exist_ok=True)
    if subphone_feat_type is not None:
        os.makedirs(os.path.join(out_dir, 'counters'), exist_ok=True)

    # Acoustic feature directories.
    os.makedirs(os.path.join(out_dir, 'f0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'lf0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'vuv'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'sp'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'ap'), exist_ok=True)

    # Sequence length feature directories.
    os.makedirs(os.path.join(out_dir, 'dur'), exist_ok=True)
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
        lf0 = np.log(f0)

        # Often there is a small difference in number of frames between labels and vocoder features.
        durations = label.phone_durations
        n_frames = sum(durations)
        diff = n_frames - f0.shape[0]

        if diff > len(durations):
            raise ValueError("Number of label frames and vocoder frames is too different for {name}\n"
                             "\tvocoder frames {voc}\n"
                             "\tlabel frames {lab}\n"
                             "\tnumber of phones {phones}"
                             .format(name=file_id, voc=f0.shape[0], lab=n_frames, phones=len(durations)))

        # Remove excess durations if there is a shape mismatch.
        if diff > 0:
            # Remove 1 frame from each phone's duration starting at the end of the sequence.
            durations[-diff:] -= 1
            n_frames = f0.shape[0]

        assert n_frames == sum(durations)

        make_feature_path = lambda name: os.path.join(out_dir, name, '{}.{}'.format(file_id, name))

        # Save linguistic features in binary .npy files.
        file_io.save_bin(numerical_labels, make_feature_path('lab'))
        if subphone_feat_type is not None:
            file_io.save_bin(counter_features[:n_frames], make_feature_path('counters'))

        # Save acoustic features in binary .npy files.
        file_io.save_bin(f0[:n_frames], make_feature_path('f0'))
        file_io.save_bin(lf0[:n_frames], make_feature_path('lf0'))
        file_io.save_bin(vuv[:n_frames], make_feature_path('vuv'))
        file_io.save_bin(sp[:n_frames], make_feature_path('sp'))
        file_io.save_bin(ap[:n_frames], make_feature_path('ap'))

        # Save sequence length features in text files.
        file_io.save_txt(durations, make_feature_path('dur'))
        file_io.save_txt(n_frames, make_feature_path('n_frames'))
        file_io.save_txt(len(label.phones), make_feature_path('n_phones'))

        # Save dimensionality of linguistic and acoustic features to text files.
        make_dim_path = lambda name: os.path.join(out_dir, '{}.dim'.format(name))

        file_io.save_txt(numerical_labels.shape[1], make_dim_path('lab'))
        if subphone_feat_type is not None:
            file_io.save_txt(counter_features.shape[1], make_dim_path('counters'))

        file_io.save_txt(f0.shape[1], make_dim_path('f0'))
        file_io.save_txt(lf0.shape[1], make_dim_path('lf0'))
        file_io.save_txt(vuv.shape[1], make_dim_path('vuv'))
        file_io.save_txt(sp.shape[1], make_dim_path('sp'))
        file_io.save_txt(ap.shape[1], make_dim_path('ap'))

    save_lab_and_wav_to_files(file_ids)

    if calc_mvn:
        calclate_mvn_parameters(out_dir, 'dur', id_list=id_list, is_npy=False)
        calclate_mvn_parameters(out_dir, 'f0', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'lf0', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'vuv', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'sp', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'ap', id_list=id_list, dtype=np.float32)


def process_lab_files(lab_dir, id_list, out_dir, state_level, question_file, subphone_feat_type, calc_mvn):
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

    if calc_mvn:
        calclate_mvn_parameters(out_dir, 'dur', id_list=id_list, is_npy=False)


def process_wav_files(wav_dir, id_list, out_dir, calc_mvn):
    """Processes wave files in id_list, saves the vocoder features to binary numpy files.

    Args:
        wav_dir (str): Directory containing the wave files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        calc_mvn (bool): If True, mean-variance normalisation parameters are calculated for acoustic features.
        """
    file_ids = utils.get_file_ids(wav_dir, id_list)

    os.makedirs(os.path.join(out_dir, 'f0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'lf0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'vuv'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'sp'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'ap'), exist_ok=True)

    @utils.multithread
    def save_wav_to_files(file_id):
        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav = wav_features.Wav(wav_path)

        f0, vuv, sp, ap = wav.extract_features()

        file_io.save_bin(f0, os.path.join(out_dir, 'f0', '{}.f0'.format(file_id)))
        file_io.save_bin(np.log(f0), os.path.join(out_dir, 'lf0', '{}.lf0'.format(file_id)))
        file_io.save_bin(vuv, os.path.join(out_dir, 'vuv', '{}.vuv'.format(file_id)))
        file_io.save_bin(sp, os.path.join(out_dir, 'sp', '{}.sp'.format(file_id)))
        file_io.save_bin(ap, os.path.join(out_dir, 'ap', '{}.ap'.format(file_id)))

    save_wav_to_files(file_ids)

    if calc_mvn:
        calclate_mvn_parameters(out_dir, 'f0', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'lf0', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'vuv', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'sp', id_list=id_list, dtype=np.float32)
        calclate_mvn_parameters(out_dir, 'ap', id_list=id_list, dtype=np.float32)


def calclate_mvn_parameters(data_dir, feat_name, id_list=None, is_npy=True, feat_dim=None, dtype=np.float32):
    """Calculates the mean-variance normalisation statistics from a directory of features.

    Args:
        data_dir (str): Root directory containing folders of features.
        feat_name (str): Name of the feature to be normalised.
        feat_dim (int): Dimensionality of the feature, required for loading from a binary file.
        id_list (str): List of file names to process.
        is_npy (bool): If True uses `file_io.load_bin`, otherwise uses `file_io.load_txt` to load each file.
        mvn_file_path (str): File to save the mean-variance normalisation parameters to.
        mvn_keys (list<str>): Names of the features to calculate mean-variance normalisation parameters for.
    """
    feat_dir = os.path.join(data_dir, feat_name)
    file_ids = utils.get_file_ids(feat_dir, id_list)

    if feat_dim is None:
        if is_npy:
            feat_dim_path = os.path.join(data_dir, '{}.dim'.format(feat_name))
            feat_dim = file_io.load_txt(feat_dim_path).item()
        else:
            feat_ex_path = os.path.join(feat_dir, '{}.{}'.format(file_ids[0], feat_name))
            feat_dim = file_io.load_txt(feat_ex_path).shape[1]

    sums = np.zeros(feat_dim, dtype=dtype)
    sum_squares = np.zeros(feat_dim, dtype=dtype)
    counts = np.zeros(feat_dim, dtype=np.int32)

    for file_id in file_ids:
        file_path = os.path.join(feat_dir, '{}.{}'.format(file_id, feat_name))

        if is_npy:
            feature = file_io.load_bin(file_path, feat_dim=feat_dim, dtype=dtype)
        else:
            feature = file_io.load_txt(file_path)

        sums += np.sum(feature, axis=0)
        sum_squares += np.sum(feature ** 2, axis=0)
        counts += feature.shape[0]

    counts = counts.astype(dtype)
    mean = sums / counts
    variance = (sum_squares - (sums ** 2) / counts) / counts
    std_dev = np.sqrt(variance)

    mvn_params = {
        'mean': mean.tolist(),
        'std_dev': std_dev.tolist()
    }

    mvn_file_path = os.path.join(data_dir, '{}_mvn.json'.format(feat_name))
    file_io.save_json(mvn_params, mvn_file_path)

    return mean, std_dev


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract duration information from forced alignment label files.")
    add_arguments(parser)
    args = parser.parse_args()

    if args.lab_dir and args.wav_dir:
        process_files(args.lab_dir, args.wav_dir, args.id_list, args.out_dir, args.state_level,
                      args.question_file, args.subphone_feat_type, args.auto_calc_mvn)

    elif args.lab_dir:
        process_lab_files(args.lab_dir, args.id_list, args.out_dir, args.state_level,
                          args.question_file, args.subphone_feat_type, args.auto_calc_mvn)

    elif args.wav_dir:
        process_wav_files(args.wav_dir, args.id_list, args.out_dir, args.auto_calc_mvn)

    elif args.feat_name:
        calclate_mvn_parameters(args.out_dir, args.feat_name, id_list=args.id_list,
                                is_npy=not args.file_is_txt, feat_dim=args.feat_dim)


if __name__ == "__main__":
    main()

