import argparse
import numpy as np
import os

from tts_data_tools import file_io
from tts_data_tools.utils import get_file_ids
from tts_data_tools.wav_gen import utils

from tts_data_tools.scripts.mean_variance_normalisation import process as process_mvn

import pyworld

WORLD_UNVOICED_VALUE = 0.


def add_arguments(parser):
    parser.add_argument("--wav_dir", action="store", dest="wav_dir", type=str, required=True,
                        help="Directory of the wave files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--calculate_normalisation", action="store_true", dest="calculate_normalisation", default=False,
                        help="Whether to automatically calculate MVN parameters after extracting F0.")
    parser.add_argument("--normalisation_of_deltas", action="store_true", dest="normalisation_of_deltas", default=False,
                        help="Also calculate the MVN parameters for the delta and delta delta features.")


def extract_vuv(f0):
    return utils.extract_vuv(f0, WORLD_UNVOICED_VALUE)


def basic_analysis(wav, sample_rate):
    nbits = wav.itemsize * 8
    int_ceiling = 2 ** (nbits - 1)
    float_data = wav.astype(np.float64) / int_ceiling
    f0, smoothed_spectrogram, aperiodicity = pyworld.wav2world(float_data, sample_rate)

    f0 = f0.reshape((-1, 1))
    return f0, smoothed_spectrogram, aperiodicity


def analysis(wav, sample_rate):
    """Extracts vocoder features using WORLD.

    Note VUV in F0 is represented using 0.0

    Returns:
        (np.ndarray[n_frames, 1]): interpolated fundamental frequency,
        (np.ndarray[n_frames, 1]): voiced-unvoiced flags,
        (np.ndarray[n_frames, sp_dim]): smoothed spectrogram,
        (np.ndarray[n_frames, ap_dim]): aperiodicity.
    """
    f0, smoothed_spectrogram, aperiodicity = basic_analysis(wav, sample_rate)

    vuv = extract_vuv(f0)
    f0_interpolated = utils.interpolate(f0, vuv)

    return f0_interpolated, vuv, smoothed_spectrogram, aperiodicity


def synthesis(f0, vuv, sp, ap, sample_rate):
    f0 = f0 * vuv

    f0 = f0.astype(np.float64)
    sp = sp.astype(np.float64)
    ap = ap.astype(np.float64)

    return pyworld.synthesize(f0, sp, ap, sample_rate)


def process(wav_dir, id_list, out_dir, calculate_normalisation, normalisation_of_deltas):
    """Processes wav files in id_list, saves the log-F0 and MVN parameters to files.

    Args:
        wav_dir (str): Directory containing the wav files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        calculate_normalisation (bool): Whether to automatically calculate MVN parameters after extracting F0.
        normalisation_of_deltas (bool): Also calculate the MVN parameters for the delta and delta delta features.
    """
    file_ids = get_file_ids(wav_dir, id_list)

    for file_id in file_ids:
        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav, sample_rate = file_io.load_wav(wav_path)

        f0, vuv, sp, ap = analysis(wav, sample_rate)
        lf0 = np.log(f0)

        file_io.save_bin(lf0, os.path.join(out_dir, 'lf0', '{}.lf0'.format(file_id)))
        file_io.save_bin(vuv, os.path.join(out_dir, 'vuv', '{}.vuv'.format(file_id)))
        file_io.save_bin(sp, os.path.join(out_dir, 'sp', '{}.sp'.format(file_id)))
        file_io.save_bin(ap, os.path.join(out_dir, 'ap', '{}.ap'.format(file_id)))

    if calculate_normalisation:
        process_mvn(out_dir, 'lf0', id_list=id_list, deltas=normalisation_of_deltas)
        process_mvn(out_dir, 'sp', id_list=id_list, deltas=normalisation_of_deltas)
        process_mvn(out_dir, 'ap', id_list=id_list, deltas=normalisation_of_deltas)


def main():
    parser = argparse.ArgumentParser(
        description="Extracts log-F0, V/UV, smoothed spectrogram, and aperiodicity from wavfiles using WORLD.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.wav_dir, args.id_list, args.out_dir, args.calculate_normalisation, args.normalisation_of_deltas)


if __name__ == "__main__":
    main()

