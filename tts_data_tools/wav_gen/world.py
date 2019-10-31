import argparse
import numpy as np
import os

from tts_data_tools import file_io
from tts_data_tools.utils import get_file_ids, make_dirs
from tts_data_tools.wav_gen import utils

from tts_data_tools.scripts.mean_variance_normalisation import process as process_mvn

import pysptk
import pyworld

UNVOICED_VALUE = 0.


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
    return utils.extract_vuv(f0, UNVOICED_VALUE)


def basic_analysis(wav, sample_rate):
    nbits = wav.itemsize * 8
    int_ceiling = 2 ** (nbits - 1)
    float_data = wav.astype(np.float64) / int_ceiling

    f0, smoothed_spectrogram, aperiodicity = pyworld.wav2world(float_data, sample_rate)

    f0 = f0.reshape((-1, 1))
    return f0, smoothed_spectrogram, aperiodicity


def freq_to_mcep(mag_spec, sample_rate, dims=60):
    r"""Convert from magnitude frequency space to mel-cepstral space.

    We use mel-cepstrum (i.e. mel-generalised with :math:`\gamma = 0`) as we do not make assumptions about the SNR.
    """
    mag_spec = mag_spec.astype(np.float64)

    # Convert float to signed-int16 domain.
    data_16bit = mag_spec * 2. ** 15

    # maxiter=0, etype=1, eps=1e-8, min_det=0.
    mcep = pysptk.mcep(data_16bit, order=dims - 1, alpha=utils.ALPHA[sample_rate], itype=3)
    return mcep


def mcep_to_freq(mcep, sample_rate):
    r"""Convert from mel-cepstral space to magnitude frequency space."""
    mcep = mcep.astype(np.float64)

    log_freq_16bit = pysptk.mgc2sp(
        mcep, alpha=utils.ALPHA[sample_rate], gamma=0., fftlen=utils.FRAME_LENGTH[sample_rate]).real

    # Convert signed-int16 to float.
    mag_spec = np.exp(log_freq_16bit) / 2. ** 15
    return mag_spec


def analysis(wav, sample_rate, mcep_dims=60, bap_dims=5):
    """Extracts vocoder features using WORLD.

    Note VUV in F0 is represented using 0.0

    Returns:
        (np.ndarray[n_frames, 1]): interpolated fundamental frequency,
        (np.ndarray[n_frames, 1]): voiced-unvoiced flags,
        (np.ndarray[n_frames, dims]): mel cepstrum,
        (np.ndarray[n_frames, dims]): band aperiodicity.
    """
    f0, smoothed_spectrogram, aperiodicity = basic_analysis(wav, sample_rate)

    vuv = extract_vuv(f0)
    f0_interpolated = utils.interpolate(f0, vuv)

    mel_cepstrum = freq_to_mcep(smoothed_spectrogram, sample_rate, dims=mcep_dims)
    band_aperiodicity = freq_to_mcep(aperiodicity, sample_rate, dims=bap_dims)

    return f0_interpolated, vuv, mel_cepstrum, band_aperiodicity


def synthesis(f0, vuv, mcep, bap, sample_rate):
    f0 = f0 * vuv
    f0 = f0.squeeze()

    sp = mcep_to_freq(mcep, sample_rate)
    ap = mcep_to_freq(bap, sample_rate)

    f0 = f0.astype(np.float64)
    sp = sp.astype(np.float64)
    ap = ap.astype(np.float64)

    return pyworld.synthesize(f0, sp, ap, sample_rate)


def process(wav_dir, id_list, out_dir, calculate_normalisation, normalisation_of_deltas):
    """Processes wav files in id_list, saves the log-F0 and MVN parameters to files and re-synthesises the speech.

    Args:
        wav_dir (str): Directory containing the wav files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        calculate_normalisation (bool): Whether to automatically calculate MVN parameters after extracting F0.
        normalisation_of_deltas (bool): Also calculate the MVN parameters for the delta and delta delta features.
    """
    file_ids = get_file_ids(id_list=id_list)

    make_dirs(os.path.join(out_dir, 'lf0'), file_ids)
    make_dirs(os.path.join(out_dir, 'vuv'), file_ids)
    make_dirs(os.path.join(out_dir, 'mcep'), file_ids)
    make_dirs(os.path.join(out_dir, 'bap'), file_ids)
    make_dirs(os.path.join(out_dir, 'wav_synth'), file_ids)

    for file_id in file_ids:
        wav_path = os.path.join(wav_dir, '{}.wav'.format(file_id))
        wav, sample_rate = file_io.load_wav(wav_path)

        f0, vuv, mcep, bap = analysis(wav, sample_rate)
        lf0 = np.log(f0)

        wav_synth = synthesis(f0, vuv, mcep, bap, sample_rate)

        file_io.save_bin(lf0, os.path.join(out_dir, 'lf0', file_id))
        file_io.save_bin(vuv, os.path.join(out_dir, 'vuv', file_id))
        file_io.save_bin(mcep, os.path.join(out_dir, 'mcep', file_id))
        file_io.save_bin(bap, os.path.join(out_dir, 'bap', file_id))
        file_io.save_wav(wav_synth, os.path.join(out_dir, 'wav_synth', f'{file_id}.wav'), sample_rate)

    if calculate_normalisation:
        process_mvn(out_dir, 'lf0', id_list=id_list, deltas=normalisation_of_deltas, out_dir=out_dir)
        process_mvn(out_dir, 'mcep', id_list=id_list, deltas=normalisation_of_deltas, out_dir=out_dir)
        process_mvn(out_dir, 'bap', id_list=id_list, deltas=normalisation_of_deltas, out_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Extracts log-F0, V/UV, smoothed spectrogram, and aperiodicity from wavfiles using WORLD.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.wav_dir, args.id_list, args.out_dir, args.calculate_normalisation, args.normalisation_of_deltas)


if __name__ == "__main__":
    main()

