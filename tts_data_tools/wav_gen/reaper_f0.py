import argparse

import pyreaper

from tts_data_tools.wav_gen import utils

REAPER_UNVOICED_VALUE = -1.


def add_arguments(parser):
    pass


def extract_vuv(f0):
    return utils.extract_vuv(f0, REAPER_UNVOICED_VALUE)


def basic_analysis(wav, sample_rate):
    pm_times, pm, f0_times, f0, corr = pyreaper.reaper(wav, sample_rate)
    f0 = f0.reshape((-1, 1))

    return f0


def analysis(wav, sample_rate):
    """Extracts vocoder features using REAPER.

    Note VUV in F0 is represented using 0.0

    Returns:
        (np.ndarray[n_frames, 1]): interpolated fundamental frequency,
        (np.ndarray[n_frames, 1]): voiced-unvoiced flags.
    """
    f0 = basic_analysis(wav, sample_rate)
    vuv = extract_vuv(f0)
    f0_interpolated = utils.interpolate(f0, vuv)

    return f0_interpolated, vuv


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
    f0, vuv = analysis(wav, sample_rate)

    save_bin(f0, '{}.f0'.format(args.out_file))
    save_bin(vuv, '{}.vuv'.format(args.out_file))


if __name__ == "__main__":
    main()

