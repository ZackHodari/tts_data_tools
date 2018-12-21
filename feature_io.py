"""Handles loading and analysing wav files.

Usage:
    python feature_io.py --wav_file FILE"""

import argparse

import numpy as np
from scipy.io import wavfile

import pyworld
import pyreaper
import pysptk

from file_io import save_bin


def add_arguments(parser):
    pass


def load_wav(file_path):
    """Loads wave data from wavfile.

    Args:
        file_path (str): File to load from.

    Returns:
        (np.ndarray) Waveform samples,
        (int) Sample rate of waveform."""
    sample_rate, data = wavfile.read(pysptk.util.example_audio_file())
    return data, sample_rate


def save_wav(file_path, data, sample_rate):
    """Saves wave data to wavfile.

    Args:
        data (np.ndarray): Waveform samples.
        file_path (str): File to save to."""
    wavfile.write(file_path, sample_rate, data)


class Wav(object):
    """Container for waveforms, allows for feature extraction."""
    def __init__(self, file_path):
        """...

        :param file_path:
        """
        self.data, self.sample_rate = load_wav(file_path)

    def reaper(self):
        """Extracts f0 using REAPER.

        :return: fundamental frequency.
        """
        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(self.data, self.sample_rate)
        return f0

    def world(self):
        """Extracts vocoder features using WORLD.

        :return: fundamental frequency, smoothed spectrogram, aperiodicity.
        """
        nbits = self.data.itemsize * 8
        int_ceiling = 2 ** (nbits - 1)
        float_data = self.data.astype(np.float64) / int_ceiling
        f0, smoothed_spectrogram, aperiodicity = pyworld.wav2world(float_data, self.sample_rate)
        return f0, smoothed_spectrogram, aperiodicity

    def extract_features(self):
        """Extracts REAPER's f0 and WORLDS segmental features. Ensures they are the same number of frames.

        :return: fundamental frequency, smoothed spectrogram, aperiodicity.
        """
        f0 = self.reaper()
        f0_world, sp, ap = self.world()

        # REAPER marks frames using their left edge, while WORLD uses their centre. This leads to 2 frames that REAPER
        # does not capture. Additionally, framing scheme differences also lead to fewer frames at the end in REAPER.
        assert f0.shape[0] <= f0_world.shape[0] - 2, (
            "We expect REAPER's f0 estimate to be at least 2 frames shorter that WORLD's f0,"
            "got len(f0_REAPER) = {}, len(f0_WORLD) = {}".format(f0.shape[0], f0_world.shape[0]))

        diff = f0_world.shape[0] - f0.shape[0]
        pad_start = np.tile(f0[0], 2)
        pad_end = np.tile(f0[-1], diff - 2)

        f0 = np.concatenate((pad_start, f0, pad_end))

        num_frames = f0.shape[0]
        f0_dim = 1
        sp_dim = sp.shape[1]
        ap_dim = ap.shape[1]
        total_dim = f0_dim + sp_dim + ap_dim

        print("Vocoder features created: {} frames; f0 dimensionality = {}; sp dimensionality = {}; "
              "ap dimensionality = {}; and {} total features.".format(num_frames, f0_dim, sp_dim, ap_dim, total_dim))
        return f0, sp, ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to load wav files.")
    parser.add_argument("--wav_file", action="store", dest="wav_file", type=str, required=True,
                        help="File path of the wavfile to be vocoded.")
    parser.add_argument("--out_file", action="store", dest="out_file", type=str, required=True,
                        help="File path (without file extension) to save the vocoder features to.")
    add_arguments(parser)
    args = parser.parse_args()

    wav = Wav(args.wav_file)

    f0, sp, ap = wav.extract_features()
    save_bin(f0, '{}.f0'.format(args.out_file))
    save_bin(sp, '{}.sp'.format(args.out_file))
    save_bin(ap, '{}.ap'.format(args.out_file))

