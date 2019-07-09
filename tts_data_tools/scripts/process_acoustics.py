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

from tts_data_tools import file_io
from tts_data_tools import utils
from tts_data_tools.wav_gen import world_with_reaper_f0

from tts_data_tools.scripts.mean_variance_normalisation import calculate_mvn_parameters
from tts_data_tools.scripts.save_features import save_lf0, save_vuv, save_sp, save_ap


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


def extract_from_waveforms(file_ids, wav_dir):

    @utils.multithread(_wav_dir=wav_dir)
    def extract(file_id, _wav_dir):
        wav_path = os.path.join(_wav_dir, '{}.wav'.format(file_id))
        wav, sample_rate = file_io.load_wav(wav_path)

        f0, vuv, sp, ap = world_with_reaper_f0.analysis(wav, sample_rate)
        lf0 = np.log(f0)

        return lf0, vuv, sp, ap

    return zip(*extract(file_ids))


def process(wav_dir, id_list, out_dir, calculate_normalisation, normalisation_of_deltas):
    """Processes wav files in id_list, saves the log-F0 and MVN parameters to files.

    Args:
        wav_dir (str): Directory containing the wav files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        calculate_normalisation (bool): Whether to automatically calculate MVN parameters after extracting F0.
        normalisation_of_deltas (bool): Also calculate the MVN parameters for the delta and delta delta features.
        """
    file_ids = utils.get_file_ids(wav_dir, id_list)

    lf0_list, vuv_list, sp_list, ap_list = extract_from_waveforms(file_ids, wav_dir)

    save_lf0(file_ids, lf0_list, out_dir)
    save_vuv(file_ids, vuv_list, out_dir)
    save_sp(file_ids, sp_list, out_dir)
    save_ap(file_ids, ap_list, out_dir)

    if calculate_normalisation:
        calculate_mvn_parameters(lf0_list, out_dir, feat_name='lf0', deltas=normalisation_of_deltas)
        calculate_mvn_parameters(sp_list, out_dir, feat_name='sp', deltas=normalisation_of_deltas)
        calculate_mvn_parameters(ap_list, out_dir, feat_name='ap', deltas=normalisation_of_deltas)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract REAPER log-F0 and WORLD spectral and aperiodic parameters from wav files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.wav_dir, args.id_list, args.out_dir, args.calculate_normalisation, args.normalisation_of_deltas)


if __name__ == "__main__":
    main()

