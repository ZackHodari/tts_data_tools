import argparse

import numpy as np

import pyworld

from tts_data_tools.wav_gen import utils

WORLD_UNVOICED_VALUE = 0.


def add_arguments(parser):
    pass


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


def main():
    from tts_data_tools.file_io import save_bin, load_wav

    parser = argparse.ArgumentParser(description="Script to load wav files.")
    parser.add_argument("--wav_file", action="store", dest="wav_file", type=str, required=True,
                        help="File path of the wavfile to be vocoded.")
    parser.add_argument("--out_file", action="store", dest="out_file", type=str, required=True,
                        help="File path (without file extension) to save the vocoder features to.")
    add_arguments(parser)
    args = parser.parse_args()

    wav, sample_rate = load_wav(args.wav_file)
    f0, vuv, sp, ap = analysis(wav, sample_rate)

    save_bin(f0, '{}.f0'.format(args.out_file))
    save_bin(vuv, '{}.vuv'.format(args.out_file))
    save_bin(sp, '{}.sp'.format(args.out_file))
    save_bin(ap, '{}.ap'.format(args.out_file))


if __name__ == "__main__":
    main()

