"""Uses pitch, duration, and intensity to automatically detect vowel prominences, using per-vowel outliers.

Usage:
    python detect_prominence.py \
        [--lab_dir DIR] [--state_level] \
        [--wav_dir DIR] \
        [--id_list FILE] \
        --out_dir DIR
"""

import argparse
from collections import namedtuple
import numpy as np
from operator import attrgetter
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from tts_data_tools import file_io
from tts_data_tools import utils


Phone = namedtuple('Phone', ('file_id', 'idx', 'vowel', 'start_idx', 'end_idx', 'duration', 'lf0', 'intensity'))


def add_arguments(parser):
    parser.add_argument("--data_dir", action="store", dest="data_dir", type=str, required=True,
                        help="Directory containing the directories of data to be clustered.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--phone_set", action="store", dest="phone_set", type=str, required=True,
                        help="Phone-set from which to take the list of vowels.")
    parser.add_argument("--method", action="store", dest="method", type=str, default='k_means',
                        help="Method by which to determine prominent vs. non-prominent vowels.")
    parser.add_argument("--verbose", action="store_true", dest="verbose", default=False,
                        help="Print out per vowel cluster information.")


def get_vowel_and_feature(file_id, phone_durations, phones, durations, lf0, intensity, vowel_list):
    cumulative_duration = np.concatenate(([0], np.cumsum(phone_durations)))

    for i, phone in enumerate(phones):

        if phone in vowel_list:

            start_idx = cumulative_duration[i]
            end_idx = cumulative_duration[i+1]

            yield Phone(file_id, i, phone, start_idx, end_idx,
                          durations[i], lf0[start_idx:end_idx], intensity[start_idx:end_idx])


def insort(a, x, lo=0, hi=None, key=None):
    if key is None:
        key = lambda e: e

    # we avoid computing key(x) in each iteration
    x_value = key(x)

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)

    while lo < hi:
        mid = (lo+hi)//2
        if x_value < key(a[mid]):
            hi = mid
        else:
            lo = mid+1

    a.insert(lo, x)


def k_means_prominence(all_vowel_features, file_ids, verbose=False):
    """Determines which vowels are prominent based on K-means clustering, returns prominent vowels per sentence.

    Args:
        all_vowel_features (dict[str, list[Phone]]): Dictionary containing vowel instances split by
            vowel identity. Vowel instances are stored as a `namedtuple` called `Phone`.
        file_ids (list[str]): Basenames of files.
        verbose (bool): Print out per vowel cluster information.

    Returns:
        prominent_vowels (dict[str, list[Phone]]): Dictionary containing list of vowel instances split by file_id.
    """
    prominent_vowels = {file_id: [] for file_id in file_ids}

    for vowel, vowel_features in all_vowel_features.items():
        if len(vowel_features) == 0:
            if verbose:
                print(f'\n\n{vowel} - No data')
            continue

        dur = np.array([vowel.duration for vowel in vowel_features])
        lf0 = np.array([np.mean(vowel.lf0, axis=0) for vowel in vowel_features])[:, None]
        intensity = np.array([np.mean(vowel.intensity, axis=0) for vowel in vowel_features])[:, None]

        mean_dur = np.mean(dur, axis=0)
        mean_lf0 = np.mean(lf0, axis=0)
        mean_intensity = np.mean(intensity, axis=0)

        std_dev_dur = np.std(dur, axis=0)
        std_dev_lf0 = np.std(lf0, axis=0)
        std_dev_intensity = np.std(intensity, axis=0)

        norm_dur = (dur - mean_dur) / std_dev_dur
        norm_lf0 = (lf0 - mean_lf0) / std_dev_lf0
        norm_intensity = (intensity - mean_intensity) / std_dev_intensity

        kmeans_dur = KMeans(n_clusters=2, random_state=1234567890).fit(norm_dur)
        kmeans_lf0 = KMeans(n_clusters=3, random_state=1234567890).fit(norm_lf0)
        kmeans_intensity = KMeans(n_clusters=2, random_state=1234567890).fit(norm_intensity)

        dur_labels = kmeans_dur.labels_
        lf0_labels = kmeans_lf0.labels_
        intensity_labels = kmeans_intensity.labels_

        dur_centers = kmeans_dur.cluster_centers_ * std_dev_dur + mean_dur
        lf0_centers = np.exp(kmeans_lf0.cluster_centers_ * std_dev_lf0 + mean_lf0)
        intensity_centers = kmeans_intensity.cluster_centers_ * std_dev_intensity + mean_intensity

        prominent_dur_cluster = np.argmax(dur_centers)
        prominent_lf0_cluster = [np.argmin(lf0_centers), np.argmax(lf0_centers)]
        prominent_intensity_cluster = np.argmax(intensity_centers)

        if verbose:
            print(f'\n\n{vowel} dur {mean_dur}')
            for i in range(2):
                print(f'{dur_centers[i].item():.4f} {dur_labels.tolist().count(i)}')

            print(f'\n{vowel} lf0 {np.exp(mean_lf0)}')
            for i in range(3):
                print(f'{lf0_centers[i].item():.4f} {lf0_labels.tolist().count(i)}')

            print(f'\n{vowel} intensity {mean_intensity}')
            for i in range(2):
                print(f'{intensity_centers[i].item():.4f} {intensity_labels.tolist().count(i)}')

        for i, vowel in enumerate(vowel_features):

            if dur_labels[i] == prominent_dur_cluster \
                    and lf0_labels[i] in prominent_lf0_cluster \
                    and intensity_labels[i] == prominent_intensity_cluster:

                insort(prominent_vowels[vowel.file_id], vowel, key=attrgetter('idx'))

    return prominent_vowels


def gmm_prominence(all_vowel_features, file_ids, verbose=False):
    """Determines which vowels are prominent based on GMM fit, returns prominent vowels per sentence.

    Args:
        all_vowel_features (dict[str, list[Phone]]): Dictionary containing vowel instances split by
            vowel identity. Vowel instances are stored as a `namedtuple` called `Phone`.
        file_ids (list[str]): Basenames of files.
        verbose (bool): Print out per vowel cluster information.

    Returns:
        prominent_vowels (dict[str, list[Phone]]): Dictionary containing list of vowel instances split by file_id.
    """
    prominent_vowels = {file_id: [] for file_id in file_ids}

    for vowel, vowel_features in all_vowel_features.items():
        if len(vowel_features) == 0:
            if verbose:
                print(f'\n\n{vowel} - No data')
            continue

        dur = np.array([vowel.duration for vowel in vowel_features])
        lf0 = np.array([np.mean(vowel.lf0, axis=0) for vowel in vowel_features])[:, None]
        intensity = np.array([np.mean(vowel.intensity, axis=0) for vowel in vowel_features])[:, None]

        mean_dur = np.mean(dur, axis=0)
        mean_lf0 = np.mean(lf0, axis=0)
        mean_intensity = np.mean(intensity, axis=0)

        std_dev_dur = np.std(dur, axis=0)
        std_dev_lf0 = np.std(lf0, axis=0)
        std_dev_intensity = np.std(intensity, axis=0)

        norm_dur = (dur - mean_dur) / std_dev_dur
        norm_lf0 = (lf0 - mean_lf0) / std_dev_lf0
        norm_intensity = (intensity - mean_intensity) / std_dev_intensity

        GMM_dur = GaussianMixture(n_components=2, random_state=1234567890).fit(norm_dur)
        GMM_lf0 = GaussianMixture(n_components=3, random_state=1234567890).fit(norm_lf0)
        GMM_intensity = GaussianMixture(n_components=2, random_state=1234567890).fit(norm_intensity)

        dur_labels = GMM_dur.predict(norm_dur)
        lf0_labels = GMM_lf0.predict(norm_lf0)
        intensity_labels = GMM_intensity.predict(norm_intensity)

        dur_centers = GMM_dur.means_ * std_dev_dur + mean_dur
        lf0_centers = np.exp(GMM_lf0.means_ * std_dev_lf0 + mean_lf0)
        intensity_centers = GMM_intensity.means_ * std_dev_intensity + mean_intensity

        prominent_dur_cluster = np.argmax(dur_centers)
        prominent_lf0_cluster = [np.argmin(lf0_centers), np.argmax(lf0_centers)]
        prominent_intensity_cluster = np.argmax(intensity_centers)

        if verbose:
            print(f'\n\n{vowel} dur {mean_dur}')
            for i in range(2):
                print(f'{dur_centers[i].item():.4f} {dur_labels.tolist().count(i)}')

            print(f'\n{vowel} lf0 {np.exp(mean_lf0)}')
            for i in range(3):
                print(f'{lf0_centers[i].item():.4f} {lf0_labels.tolist().count(i)}')

            print(f'\n{vowel} intensity {mean_intensity}')
            for i in range(2):
                print(f'{intensity_centers[i].item():.4f} {intensity_labels.tolist().count(i)}')

        for i, vowel in enumerate(vowel_features):

            if dur_labels[i] == prominent_dur_cluster \
                    and lf0_labels[i] in prominent_lf0_cluster \
                    and intensity_labels[i] == prominent_intensity_cluster:

                insort(prominent_vowels[vowel.file_id], vowel, key=attrgetter('idx'))

    return prominent_vowels


def threshold_prominence(all_vowel_features, file_ids, verbose=False):
    """Determines which vowels are prominent based on manually set thresholds, returns prominent vowels per sentence.

    Args:
        all_vowel_features (dict[str, list[Phone]]): Dictionary containing vowel instances split by
            vowel identity. Vowel instances are stored as a `namedtuple` called `Phone`.
        file_ids (list[str]): Basenames of files.
        verbose (bool): Print out per vowel cluster information.

    Returns:
        prominent_vowels (dict[str, list[Phone]]): Dictionary containing list of vowel instances split by file_id.
    """
    prominent_vowels = {file_id: [] for file_id in file_ids}

    for vowel, vowel_features in all_vowel_features.items():
        if len(vowel_features) == 0:
            if verbose:
                print(f'\n\n{vowel} - No data')
            continue

        dur = np.array([vowel.duration for vowel in vowel_features])
        lf0 = np.array([np.mean(vowel.lf0) for vowel in vowel_features])
        intensity = np.array([np.mean(vowel.intensity) for vowel in vowel_features])

        mean_dur = np.mean(dur, axis=0)
        mean_lf0 = np.mean(lf0, axis=0)
        mean_intensity = np.mean(intensity, axis=0)

        std_dev_dur = np.std(dur, axis=0)
        std_dev_lf0 = np.std(lf0, axis=0)
        std_dev_intensity = np.std(intensity, axis=0)

        for i, vowel in enumerate(vowel_features):

            dur_full = dur[i] > mean_dur + std_dev_dur
            dur_partial = dur[i] > mean_dur + 0.5 * std_dev_dur

            lf0_lower_full = lf0[i] < mean_lf0 - std_dev_lf0
            lf0_lower_partial = lf0[i] < mean_lf0 - 0.5 * std_dev_lf0

            lf0_upper_full = lf0[i] > mean_lf0 + std_dev_lf0
            lf0_upper_partial = lf0[i] > mean_lf0 + 0.5 * std_dev_lf0

            intensity_upper_full = intensity[i] > mean_intensity + std_dev_intensity
            intensity_upper_partial = intensity[i] > mean_intensity + 0.5 * std_dev_intensity

            if any((dur_full, lf0_lower_full, lf0_upper_full, intensity_upper_full)):
                insort(prominent_vowels[vowel.file_id], vowel, key=attrgetter('idx'))

            elif dur_partial and (lf0_lower_partial or lf0_upper_partial) and intensity_upper_partial:
                insort(prominent_vowels[vowel.file_id], vowel, key=attrgetter('idx'))

    return prominent_vowels


def process(data_dir, id_list, out_dir, phone_set=None, vowel_list=None, method='k_means', verbose=False):
    """Extracts pitch, duration, and intensity. Uses per-vowel outliers to automatically detect vowel prominences.

    Args:
        data_dir (str): Directory containing the directories of features to be clustered.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        phone_set (str): Name of the phone_set from which to take vowel list.
        vowel_list (list[str]): Vowels for the phone-set used in the label files.
        method ({'k_means', 'gmm', 'threshold'}): Method by which to determine prominent vs. non-prominent vowels.
        verbose (bool): Print out per vowel information.

    Notes:
        Exactly one of phone_set or vowels must be specified.
    """
    if phone_set is None and vowel_list is None:
        raise AttributeError('Exactly one of phone_set and vowels must be given, neither were provided.')
    if phone_set is not None and vowel_list is not None:
        raise AttributeError('Exactly one of phone_set and vowels must be given, both were provided.')

    if phone_set is not None:
        if phone_set == 'radio':
            vowel_list = [
                'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay', 'eh', 'el', 'em', 'en', 'er', 'ey', 'ih',
                'ix', 'iy', 'ow', 'oy', 'uh', 'uw'
            ]
        elif phone_set == 'unilex':
            vowel_list = [
                'e', 'a', 'aa', 'aa1', 'ar', 'eh', 'ou', 'ou1', 'o', 'oo', 'or', 'ii', 'iy', 'i', '@r',
                '@', 'uh', 'u', 'uu', 'uw', 'ei', 'ei1', 'ai', 'aer', 'oi', 'ow', 'owr', 'i@', 'ir', '@@r',
                '@@r1', 'eir', 'eir1', 'ur', 'ur1'
            ]
        else:
            raise ValueError(f'Vowels for phone_set ({phone_set}) is not pre-defined.')

    if method not in ['k_means', 'gmm', 'threshold']:
        raise ValueError(f'No method {method}')

    file_ids = utils.get_file_ids(id_list=id_list)
    utils.make_dirs(os.path.join(out_dir, 'prominence_segment_dur'), file_ids)
    sentence_norm_params = {}

    # Vowel features that will be used to determine prominence.
    all_vowel_features = {vowel: [] for vowel in vowel_list}

    for file_id in tqdm(file_ids):
        phones = file_io.load_lines(os.path.join(data_dir, 'phones', f'{file_id}.txt'))
        dur = file_io.load_txt(os.path.join(data_dir, 'dur', f'{file_id}.txt'))
        lf0 = file_io.load_bin(os.path.join(data_dir, 'lf0', f'{file_id}.npy'))
        intensity = file_io.load_bin(os.path.join(data_dir, 'intensity', f'{file_id}.npy'))

        norm_params = {
            'dur': {'mean': np.mean(dur, axis=0), 'std_dev': np.std(dur, axis=0)},
            'lf0': {'mean': np.mean(lf0, axis=0), 'std_dev': np.std(lf0, axis=0)},
            'intensity': {'mean': np.mean(intensity, axis=0), 'std_dev': np.std(intensity, axis=0)},
        }
        sentence_norm_params[file_id] = norm_params

        # Normalise features within sentence.
        norm_dur = (dur - norm_params['dur']['mean']) / norm_params['dur']['std_dev']
        norm_lf0 = (lf0 - norm_params['lf0']['mean']) / norm_params['lf0']['std_dev']
        norm_intensity = (intensity - norm_params['intensity']['mean']) / norm_params['intensity']['std_dev']

        # Get all vowels for this sentence. Save the features for statistics calculation below.
        for vowel in get_vowel_and_feature(file_id, dur, phones, norm_dur, norm_lf0, norm_intensity, vowel_list):
            all_vowel_features[vowel.vowel].append(vowel)

    # Determine all prominent Phones per sentence.
    if method == 'k_means':
        prominent_vowels = k_means_prominence(all_vowel_features, file_ids, verbose=verbose)
    elif method == 'gmm':
        prominent_vowels = gmm_prominence(all_vowel_features, file_ids, verbose=verbose)
    elif method == 'threshold':
        prominent_vowels = threshold_prominence(all_vowel_features, file_ids, verbose=verbose)

    vowels_per_sentence = [len(vowel_features) for vowel_features in all_vowel_features.values()]
    prominent_vowels_per_sentence = [len(sentence_prom_vowels) for sentence_prom_vowels in prominent_vowels.values()]

    total_prominent_vowels = sum(prominent_vowels_per_sentence)
    sentences_with_0 = prominent_vowels_per_sentence.count(0)
    sentences_with_1 = prominent_vowels_per_sentence.count(1)
    sentences_with_gt_1 = len(file_ids) - sentences_with_0 - sentences_with_1

    print(f'{sum(vowels_per_sentence)} total vowels')
    print(f'{total_prominent_vowels} prominent vowels')
    print(f'{sentences_with_0} sentences without a prominent vowel')
    print(f'{sentences_with_1} sentences with one prominent vowel')
    print(f'{sentences_with_gt_1} sentences with more than 1 prominent vowels')

    # Report longest sentences without a prominence, as these will not be split into smaller units.
    n_phones_top_10 = [0] * 10
    n_seconds_top_10 = [0.] * 10
    for file_id, prominent_sentence_vowels in prominent_vowels.items():
        if len(prominent_sentence_vowels) < 2:
            n_phones = len(file_io.load_lines(os.path.join(data_dir, 'phones', f'{file_id}.txt')))
            n_seconds = 0.005 * np.sum(file_io.load_txt(os.path.join(data_dir, 'dur', f'{file_id}.txt')))

            if n_phones > n_phones_top_10[0]:
                insort(n_phones_top_10, n_phones)
                n_phones_top_10 = n_phones_top_10[1:]

            if n_seconds > n_seconds_top_10[0]:
                insort(n_seconds_top_10, n_seconds)
                n_seconds_top_10 = n_seconds_top_10[1:]

    print('Longest 10 sentences less than 2 prominent vowels:')
    for n_phones, n_seconds in zip(n_phones_top_10, n_seconds_top_10):
        print(f'\t{n_phones:>2} phones, {n_seconds:.2f} seconds')


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised prominence detector.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.data_dir, args.id_list, args.out_dir,
            phone_set=args.phone_set, method=args.method, verbose=args.verbose)


if __name__ == "__main__":
    main()

