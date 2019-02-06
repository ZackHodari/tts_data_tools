"""Handles loading and analysing wav files.

Usage:
    python wav_features.py --wav_file FILE --out_file FILE
"""

import argparse

import numpy as np

import pyworld
import pyreaper

from .file_io import save_bin, load_wav


def add_arguments(parser):
    pass


class Wav(object):
    """Container for waveforms, allows for feature extraction."""
    def __init__(self, file_path):
        """Loads waveform.

        Args:
            file_path (str): Wave file to be loaded.
        """
        self.data, self.sample_rate = load_wav(file_path)

    def reaper(self):
        """Extracts f0 using REAPER.

        Note VUV in F0 is represented using -1.0

        Returns:
            (np.ndarray[n_frames]): fundamental frequency.
        """
        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(self.data, self.sample_rate)
        return f0

    def world(self):
        """Extracts vocoder features using WORLD.

        Note VUV in F0 is represented using 0.0

        Returns:
            (np.ndarray[n_frames]): fundamental frequency,
            (np.ndarray[n_frames, sp_dim]): smoothed spectrogram,
            (np.ndarray[n_frames, ap_dim]): aperiodicity.
        """
        nbits = self.data.itemsize * 8
        int_ceiling = 2 ** (nbits - 1)
        float_data = self.data.astype(np.float64) / int_ceiling
        f0, smoothed_spectrogram, aperiodicity = pyworld.wav2world(float_data, self.sample_rate)
        return f0, smoothed_spectrogram, aperiodicity

    def interpolate(self, signal, is_voiced):
        """Linearly interpolates the signal in unvoiced regions such that there are no discontinuities.

        Args:
            signal (np.ndarray[n_frames, feat_dim]): Temporal signal.
            is_voiced (np.ndarray[n_frames]<bool>): Boolean array indicating if each frame is voiced.

        Returns:
            (np.ndarray[n_frames, feat_dim]): Interpolated signal, same shape as signal.
        """
        n_frames = signal.shape[0]
        feat_dim = signal.shape[1]

        # Initialise whether we are starting the search in voice/unvoiced.
        in_voiced_region = is_voiced[0]

        last_voiced_frame_i = None
        for i in range(n_frames):
            if is_voiced[i]:
                if not in_voiced_region:
                    # Current frame is voiced, but last frame was unvoiced.
                    # This is the first voiced frame after an unvoiced sequence, interpolate the unvoiced region.

                    # If the signal starts with an unvoiced region then `last_voiced_frame_i` will be None.
                    # Bypass interpolation and just set this first unvoiced region to the current voiced frame value.
                    if last_voiced_frame_i is None:
                        signal[:i + 1] = signal[i]

                    # Use `np.linspace` to create a interpolate a region that includes the bordering voiced frames.
                    else:
                        start_voiced_value = signal[last_voiced_frame_i]
                        end_voiced_value = signal[i]

                        unvoiced_region_length = (i + 1) - last_voiced_frame_i
                        interpolated_region = np.linspace(start_voiced_value, end_voiced_value, unvoiced_region_length)
                        interpolated_region = interpolated_region.reshape((unvoiced_region_length, feat_dim))

                        signal[last_voiced_frame_i:i + 1] = interpolated_region

                # Move pointers forward, we are waiting to find another unvoiced section.
                last_voiced_frame_i = i

            in_voiced_region = is_voiced[i]

        # If the signal ends with an unvoiced region then it would not have been caught in the loop.
        # Similar to the case with an unvoiced region at the start we can bypass the interpolation.
        if not in_voiced_region:
            signal[last_voiced_frame_i:] = signal[last_voiced_frame_i]

        return signal

    def extract_features(self):
        """Extracts REAPER's f0 and WORLDS segmental features. Ensures they are the same number of frames.

        Returns:
            (np.ndarray[n_frames, 1]): fundamental frequency,
            (np.ndarray[n_frames, sp_dim]): smoothed spectrogram,
            (np.ndarray[n_frames, ap_dim]): aperiodicity.
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

        f0 = np.concatenate((pad_start, f0, pad_end)).reshape((-1, 1))

        is_unvoiced = np.isclose(f0, -1. * np.ones_like(f0), atol=1e-6)
        is_voiced = np.logical_not(is_unvoiced)
        f0_interpolated = self.interpolate(f0, is_voiced)

        num_frames = f0_interpolated.shape[0]
        f0_dim = f0_interpolated.shape[1]
        sp_dim = sp.shape[1]
        ap_dim = ap.shape[1]
        total_dim = f0_dim + sp_dim + ap_dim

        print("Vocoder features created: {} frames; f0 dimensionality = {}; sp dimensionality = {}; "
              "ap dimensionality = {}; and {} total features.".format(num_frames, f0_dim, sp_dim, ap_dim, total_dim))
        return f0_interpolated, is_voiced, sp, ap


def main():
    parser = argparse.ArgumentParser(description="Script to load wav files.")
    parser.add_argument("--wav_file", action="store", dest="wav_file", type=str, required=True,
                        help="File path of the wavfile to be vocoded.")
    parser.add_argument("--out_file", action="store", dest="out_file", type=str, required=True,
                        help="File path (without file extension) to save the vocoder features to.")
    add_arguments(parser)
    args = parser.parse_args()

    wav = Wav(args.wav_file)

    f0, vuv, sp, ap = wav.extract_features()
    save_bin(f0, '{}.f0'.format(args.out_file))
    save_bin(vuv, '{}.vuv'.format(args.out_file))
    save_bin(sp, '{}.sp'.format(args.out_file))
    save_bin(ap, '{}.ap'.format(args.out_file))


if __name__ == "__main__":
    main()

