import argparse

import numpy as np

from tts_data_tools.wav_gen import world, reaper_f0, utils


def add_arguments(parser):
    pass


def basic_analysis(wav, sample_rate):
    f0 = reaper_f0.basic_analysis(wav, sample_rate)
    f0_world, sp, ap = world.basic_analysis(wav, sample_rate)

    # REAPER marks frames using their left edge, while WORLD uses their centre. This leads to 2 frames that REAPER
    # does not capture. Additionally, framing scheme differences also lead to fewer frames at the end in REAPER.
    diff = f0_world.shape[0] - f0.shape[0]
    assert diff >= 2, (
        "We expect REAPER's f0 estimate to be at least 2 frames shorter that WORLD's f0,"
        "got len(f0_REAPER) = {}, len(f0_WORLD) = {}".format(f0.shape[0], f0_world.shape[0]))

    pad_start = np.repeat(f0[0, np.newaxis], 2, axis=0)
    pad_end = np.repeat(f0[-1, np.newaxis], diff - 2, axis=0)

    f0 = np.concatenate((pad_start, f0, pad_end), axis=0)

    return f0, sp, ap


def analysis(wav, sample_rate):
    """Extracts REAPER's f0 and WORLDS segmental features. Ensures they are the same number of frames.

    Note VUV in F0 is represented using -1.0

    Returns:
        (np.ndarray[n_frames, 1]): interpolated fundamental frequency,
        (np.ndarray[n_frames, 1]): voiced-unvoiced flags,
        (np.ndarray[n_frames, sp_dim]): smoothed spectrogram,
        (np.ndarray[n_frames, ap_dim]): aperiodicity.
    """
    f0, sp, ap = basic_analysis(wav, sample_rate)

    vuv = reaper_f0.extract_vuv(f0)
    f0_interpolated = utils.interpolate(f0, vuv)

    return f0_interpolated, vuv, sp, ap


def synthesis(f0, vuv, sp, ap, sample_rate):
    return world.synthesis(f0, vuv, sp, ap, sample_rate)


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



