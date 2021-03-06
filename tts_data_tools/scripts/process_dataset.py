"""Extracts linguistic and acoustic features.

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
from tqdm import tqdm

from tts_data_tools import file_io
from tts_data_tools import lab_gen
from tts_data_tools import utils
from tts_data_tools.lab_gen import lab_to_feat
from tts_data_tools.wav_gen import world_with_reaper_f0

from tts_data_tools.scripts.mean_variance_normalisation import process as process_mvn
from tts_data_tools.scripts.min_max_normalisation import process as process_minmax


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
    parser.add_argument("--trim_silences", action="store_true", dest="trim_silences", default=False,
                        help="Whether to trim start and end silences from all features.")
    parser.add_argument("--calculate_normalisation", action="store_true", dest="calculate_normalisation", default=False,
                        help="Whether to automatically calculate MVN parameters after extracting F0.")
    parser.add_argument("--normalisation_of_deltas", action="store_true", dest="normalisation_of_deltas", default=False,
                        help="Also calculate the MVN parameters for the delta and delta delta features.")
    lab_gen.add_arguments(parser)


def process(lab_dir, wav_dir, id_list, out_dir,
            state_level, question_file, upsample, subphone_feat_type,
            trim_silences, calculate_normalisation, normalisation_of_deltas):
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
        trim_silences (bool): Whether to trim start and end silences from all features.
        calculate_normalisation (bool): Whether to automatically calculate MVN parameters after extracting F0.
        normalisation_of_deltas (bool): Also calculate the MVN parameters for the delta and delta delta features.
    """
    file_ids = utils.get_file_ids(id_list=id_list)

    question_set = lab_to_feat.QuestionSet(question_file)
    subphone_feature_set = lab_to_feat.SubphoneFeatureSet(subphone_feat_type)

    utils.make_dirs(os.path.join(out_dir, 'lab'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'counters'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'dur'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'phones'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'n_frames'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'n_phones'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'lf0'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'vuv'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'mcep'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'bap'), file_ids)

    for file_id in tqdm(file_ids):
        # Label processing.
        lab_path = os.path.join(lab_dir, f'{file_id}.lab')
        label = lab_to_feat.Label(lab_path, state_level)

        numerical_labels = label.extract_numerical_labels(question_set, upsample_to_frame_level=upsample)
        counter_features = label.extract_counter_features(subphone_feature_set)
        durations = label.phone_durations.reshape((-1, 1))
        phones = label.phones

        n_frames = np.sum(durations).item()
        n_phones = len(label.phones)

        # Acoustic processing.
        wav_path = os.path.join(wav_dir, f'{file_id}.wav')
        wav, sample_rate = file_io.load_wav(wav_path)

        f0, vuv, mcep, bap = world_with_reaper_f0.analysis(wav, sample_rate)
        lf0 = np.log(f0)

        # Match the number of frames between label forced-alignment and vocoder analysis.
        # Often the durations from forced alignment are a few frames longer than the vocoder features.
        diff = n_frames - f0.shape[0]
        if diff > n_phones:
            raise ValueError(f'Number of label frames and vocoder frames is too different for {file_id}\n'
                             f'\tlabel frames {n_frames}\n'
                             f'\tvocoder frames {f0.shape[0]}\n'
                             f'\tnumber of phones {n_phones}')

        # Remove excess durations if there is a shape mismatch.
        if diff > 0:
            # Remove 1 frame from each phone's duration starting at the end of the sequence.
            durations[-diff:] -= 1
            n_frames = f0.shape[0]
            print(f'Cropped {diff} frames from durations for utterance {file_id}')

        assert n_frames == np.sum(durations).item()

        trim_frame_slice = slice(0, n_frames)
        if trim_silences:

            start_phone_idx, end_phone_idx = 0, n_phones
            start_frame_idx, end_frame_idx = 0, n_frames
            if phones[0] in ['sil', '#']:
                start_phone_idx += 1
                start_frame_idx += durations[0]
            if phones[-1] in ['sil', '#']:
                end_phone_idx -= 1
                end_frame_idx -= durations[-1]

            trim_phone_slice = slice(int(start_phone_idx), int(end_phone_idx))
            trim_frame_slice = slice(int(start_frame_idx), int(end_frame_idx))

            numerical_labels = numerical_labels[trim_frame_slice if upsample else trim_phone_slice]
            durations = durations[trim_phone_slice]
            phones = phones[trim_phone_slice]

            n_frames = trim_frame_slice.stop - trim_frame_slice.start
            n_phones = trim_phone_slice.stop - trim_phone_slice.start

        counter_features = counter_features[trim_frame_slice]
        lf0 = lf0[trim_frame_slice]
        vuv = vuv[trim_frame_slice]
        mcep = mcep[trim_frame_slice]
        bap = bap[trim_frame_slice]

        file_io.save_bin(numerical_labels.astype(np.float32), os.path.join(out_dir, 'lab', f'{file_id}.npy'))
        file_io.save_bin(counter_features.astype(np.float32), os.path.join(out_dir, 'counters', f'{file_id}.npy'))
        file_io.save_txt(durations, os.path.join(out_dir, 'dur', f'{file_id}.txt'))
        file_io.save_lines(phones, os.path.join(out_dir, 'phones', f'{file_id}.txt'))

        file_io.save_txt(n_frames, os.path.join(out_dir, 'n_frames', f'{file_id}.txt'))
        file_io.save_txt(n_phones, os.path.join(out_dir, 'n_phones', f'{file_id}.txt'))

        file_io.save_bin(lf0.astype(np.float32), os.path.join(out_dir, 'lf0', f'{file_id}.npy'))
        file_io.save_bin(vuv, os.path.join(out_dir, 'vuv', f'{file_id}.npy'))
        file_io.save_bin(mcep.astype(np.float32), os.path.join(out_dir, 'mcep', f'{file_id}.npy'))
        file_io.save_bin(bap.astype(np.float32), os.path.join(out_dir, 'bap', f'{file_id}.npy'))

    if calculate_normalisation:
        process_minmax(out_dir, 'lab', id_list, out_dir=out_dir)
        process_minmax(out_dir, 'counters', id_list, out_dir=out_dir)
        process_mvn(out_dir, 'dur', is_npy=False, id_list=id_list, deltas=False, out_dir=out_dir)

        process_mvn(out_dir, 'lf0', id_list=id_list, deltas=normalisation_of_deltas, out_dir=out_dir)
        process_mvn(out_dir, 'mcep', id_list=id_list, deltas=normalisation_of_deltas, out_dir=out_dir)
        process_mvn(out_dir, 'bap', id_list=id_list, deltas=normalisation_of_deltas, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract linguistic and acoustic features ready for training an acoustic model.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.wav_dir, args.id_list, args.out_dir,
            args.state_level, args.question_file, args.upsample_to_frame_level, args.subphone_feat_type,
            args.trim_silences, args.calculate_normalisation, args.normalisation_of_deltas)


if __name__ == "__main__":
    main()

