import numpy as np
from scipy.signal import convolve2d


def compute_running_window(feature, window):
    """Computing dynamic features using a window is exactly a convolution operation."""
    # Check that the window length is odd.
    assert len(window) % 2 == 1

    # Pad the feature with the first and last frames.
    pad_len = (len(window) - 1) // 2
    padded_feature = np.concatenate((
        feature[[0] * pad_len],
        feature,
        feature[[-1] * pad_len]))

    # Ensure the window is an array and in the right shape to be used as a 1-dimensional kernel.
    kernel_1d = np.array(window).reshape(-1, 1)

    # We actually need to compute cross-correlation, not convolution, therefore we must rotate the 1-d kernel 180.
    return convolve2d(padded_feature, kernel_1d[::-1], 'valid')


def compute_deltas(feature):
    velocity = compute_running_window(feature, [-0.5, 0.0, 0.5])
    acceleration = compute_running_window(feature, [1., -2., 1.])

    return np.concatenate((feature, velocity, acceleration), axis=1)


def extract_vuv(signal, unvoiced_value):
    is_unvoiced = np.isclose(signal, unvoiced_value * np.ones_like(signal), atol=1e-6)
    is_voiced = np.logical_not(is_unvoiced)
    return is_voiced


def interpolate(signal, is_voiced):
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

