"""Uses pitch, duration, and intensity to automatically detect vowel prominences, using per-vowel outliers.

Usage:
    python detect_prominence.py \
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
# from tts_data_tools import lab_features
from tts_data_tools import utils
# from tts_data_tools.wav_gen import world_with_reaper_f0


def add_arguments(parser):
    # parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
    #                     help="Directory of the label files to be converted.")
    parser.add_argument("--wav_dir", action="store", dest="wav_dir", type=str, required=True,
                        help="Directory of the wave files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--window_size", action="store", dest="window_size", type=float, default=0.01,
                        help="Width (in seconds) of windows used to extract frame-level features.")
    parser.add_argument("--window_shift", action="store", dest="window_shift", type=float, default=0.005,
                        help="Shift (in seconds) of windows used to extract frame-level features.")
    # lab_features.add_arguments(parser)


def extract_intensity(wav, sample_rate, window_size=0.01, window_shift=0.005):
    samples_per_window = int(window_size * sample_rate)
    samples_per_shift = int(window_shift * sample_rate)

    # Convert wav to range [0, 1] (see `scipy.io.wavfile.read`).
    if np.issubdtype(wav.dtype, np.floating):
        # Waveforms loaded as floats will be in the range [-1, 1].
        wav = (wav + 1.) / 2.
    else:
        # Waveforms loaded as integers will use the full range of that data type.
        dtype = np.iinfo(wav.dtype)
        wav = (wav.astype(np.float32) - dtype.min) / (dtype.max - dtype.min)

    gain = np.log(wav)
    gain = np.concatenate((gain, np.tile(gain[-1], samples_per_window // 2)))

    # The last frame cannot contain more than 50% padding values.
    n_samples = wav.shape[0]
    last_frame_idx = n_samples - samples_per_window // 2

    n_frames = last_frame_idx // samples_per_shift + 1
    intensity = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        intensity[i] = np.mean(gain[i * samples_per_shift: i * samples_per_shift + samples_per_window])

    return intensity


# def process(lab_dir, wav_dir, id_list, out_dir, state_level=True, window_size=0.01, window_shift=0.005):
#     """Extracts phone identity, phone duration, pitch, and intensity.
#
#     Args:
#         lab_dir (str): Directory containing the label files.
#         wav_dir (str): Directory containing the wav files.
#         id_list (str): List of file basenames to process.
#         out_dir (str): Directory to save the output to.
#         state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
#         window_size (float): Width (in seconds) of windows used to extract frame-level features.
#         window_shift (float): Shift (in seconds) of windows used to extract frame-level features.
#     """
#     file_ids = utils.get_file_ids(id_list=id_list)
#
#     utils.make_dirs(os.path.join(out_dir, 'phones'), file_ids)
#     utils.make_dirs(os.path.join(out_dir, 'dur'), file_ids)
#     utils.make_dirs(os.path.join(out_dir, 'lf0'), file_ids)
#     utils.make_dirs(os.path.join(out_dir, 'intensity'), file_ids)
#
#     for file_id in tqdm(file_ids):
#         # Get phones and their durations.
#         lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
#         label = lab_features.Label(lab_path, state_level)
#
#         durations = label.phone_durations
#         phones = label.phones
#
#         # Extract pitch.
#         wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
#         wav, sample_rate = file_io.load_wav(wav_path)
#
#         f0, vuv, sp, ap = world_with_reaper_f0.analysis(wav, sample_rate)
#         lf0 = np.log(f0)[:, 0]
#
#         # Extract intensity.
#         intensity = extract_intensity(wav, sample_rate, window_size, window_shift)
#
#         file_io.save_lines(phones, os.path.join(out_dir, 'phones', f'{file_id}.txt'))
#         file_io.save_txt(durations, os.path.join(out_dir, 'dur', f'{file_id}.txt'))
#         file_io.save_bin(lf0, os.path.join(out_dir, 'lf0', file_id))
#         file_io.save_bin(intensity, os.path.join(out_dir, 'intensity', file_id))


def process(wav_dir, id_list, out_dir, window_size=0.01, window_shift=0.005):
    """Extracts phone identity, phone duration, pitch, and intensity.

    Args:
        wav_dir (str): Directory containing the wav files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        window_size (float): Width (in seconds) of windows used to extract frame-level features.
        window_shift (float): Shift (in seconds) of windows used to extract frame-level features.
    """
    file_ids = utils.get_file_ids(id_list=id_list)

    utils.make_dirs(os.path.join(out_dir, 'intensity'), file_ids)

    for file_id in tqdm(file_ids):
        # Extract intensity.
        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav, sample_rate = file_io.load_wav(wav_path)
        intensity = extract_intensity(wav, sample_rate, window_size, window_shift)

        file_io.save_bin(intensity, os.path.join(out_dir, 'intensity', file_id))


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised prominence detector.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.wav_dir, args.id_list, args.out_dir,
            window_size=args.window_size, window_shift=args.window_shift)


if __name__ == "__main__":
    main()

