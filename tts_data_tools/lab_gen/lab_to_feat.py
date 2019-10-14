"""Handles loading and modifying label files.

Usage:
    python lab_to_feat.py \
        --lab_file FILE [--state_level] \
        --question_file FILE [--subphone_feat_type STR] \
        --out_file FILE
"""

import argparse
from enum import Enum
import os
import pkg_resources
import re

import numpy as np
from scipy.stats import norm

from tts_data_tools import file_io
from tts_data_tools import lab_gen
from tts_data_tools.utils import get_file_ids

from tts_data_tools.scripts.mean_variance_normalisation import process as process_mvn
from tts_data_tools.scripts.min_max_normalisation import process as process_minmax

STATES_PER_PHONE = 5
FRAME_SHIFT_MS = 5

SubphoneFeatureTypeEnum = Enum(
    "SubphoneFeatureTypeEnum",
    ('FULL', 'MINIMAL_PHONEME', 'MINIMAL_FRAME', 'FRAME_ONLY', 'STATE_ONLY', 'UNIFORM_STATE', 'COARSE_CODING', 'NONE'))


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--question_file", action="store", dest="question_file", type=str, required=True,
                        help="File containing the '.hed' question set to query the labels with.")
    parser.add_argument("--upsample_to_frame_level", action="store_true", dest="upsample_to_frame_level", default=False,
                        help="Whether to upsample the numerical labels to frame-level.")
    parser.add_argument("--subphone_feat_type", action="store", dest="subphone_feat_type", type=str, default=None,
                        help="The type of subphone counter features.")
    parser.add_argument("--calculate_normalisation", action="store_true", dest="calculate_normalisation", default=False,
                        help="Whether to automatically calculate MVN parameters after extracting label features.")
    lab_gen.add_arguments(parser)


class QuestionSet(object):
    """Container for question set regexes.

    Attributes:
        file_path (str): Question set that was loaded.
        lines (list<str>): Lines from the question set.
        binary_regexes (list<regex>): Compiled regexes for creating one-hot features.
        numerical_regexes (list<regex>): Compiled regexes for creating numerical features.
    """
    def __init__(self, file_path):
        """Loads question set and prepares regexes for querying.

        Attributes:
            file_path (str): Question set to be loaded. Can be one of the four provided question sets;
                questions-unilex_dnn_600.hed
                questions-radio_dnn_416.hed
                questions-radio_phones_48.hed
                questions-mandarin.hed
                questions-japanese.hed
        """
        if file_path in pkg_resources.resource_listdir('tts_data_tools', os.path.join('resources', 'question_sets')):
            print(f'Using tts_data_tools resource from resources/question_sets for {file_path}')
            file_path = pkg_resources.resource_filename('tts_data_tools',
                                                        os.path.join('resources', 'question_sets', file_path))

        self.file_path = file_path
        self.lines = file_io.load_lines(self.file_path)
        # Ensure the only whitespaces are single space characters.
        self.lines = list(map(lambda l: re.sub('\s+', ' ', l), self.lines))

        self.binary_regexes, self.numerical_regexes = self.compile_questions(self.lines)

    @property
    def dim(self):
        return len(self.binary_regexes) + len(self.numerical_regexes)

    def query(self, label):
        """Queries the full-context label using the binary and numerical questions.

        Args:
            label (str): a HTS-style full-context linguistic feature label, compatible with the loaded question set.

        Returns:
            (np.ndarray): Normalised features.
        """
        binary_features = self.binary_queries(label)
        numerical_features = self.numerical_queries(label)
        return np.concatenate((binary_features, numerical_features))

    def binary_queries(self, label):
        """Queries the full-context label using the binary questions, for creating one-hot features.

        Args:
            label (str): a HTS-style full-context linguistic feature label, compatible with the loaded question set.

        Returns:
            (np.ndarray): One-hot features.
        """
        vector = np.zeros(len(self.binary_regexes), dtype=np.float32)
        for i, (name, patterns) in enumerate(self.binary_regexes):
            for pattern in patterns:
                # If any of the patterns evaluate to true then flip the bit and move to the next set of patterns.
                if pattern.search(label):
                    vector[i] = 1
                    break
        return vector

    def numerical_queries(self, label):
        """Queries the full-context label using the numerical questions, for creating numerical features.

        Args:
            label (str): a HTS-style full-context linguistic feature label, compatible with the loaded question set.

        Returns:
            (np.ndarray): Numerical features.
        """
        vector = np.zeros(len(self.numerical_regexes), dtype=np.float32)
        for i, (name, pattern) in enumerate(self.numerical_regexes):
            match = pattern.search(label)
            if match:
                vector[i] = match.group(1)
            else:
                vector[i] = -1.
        return vector

    def compile_questions(self, lines):
        """Extracts regexes from lines, converts them to compiled python regexes, and separates them by type.

        Args:
            lines (list<str>): Lines from the question set.

        Returns:
            (list<(str, regex)>): name and regex tuples, used for creating one-hot features,
            (list<(str, regex)>): name and regex tuples, used for creating numerical features.
        """
        binary_regexes = []
        numerical_regexes = []

        for line in lines:
            # Separate the space-delimited line.
            question_type, name, patterns = line.split(' ')

            # Remove the quotes from around the name.
            name = name[1:-1]

            # Remove the brackets around the patterns, and split into a list of patterns.
            patterns = patterns[1:-1].split(',')

            regexes = self.compile_question(question_type, patterns)

            if question_type == 'QS':
                binary_regexes.append((name, regexes))
            elif question_type == 'CQS':
                numerical_regexes.append((name, regexes))

        return binary_regexes, numerical_regexes

    def compile_question(self, question_type, patterns):
        """Converts a list of HTS-style questions into python regexes

        Args:
            question_type (str): Indicates if the question output will be binary ('QS') or numerical ('CQS').
            patterns (list<str>): Wildcard-based HTK question regex strings.

        Returns:
            Compiled python regex.
        """
        if question_type == 'QS':
            patterns = map(self.wildcards_to_python_regex, patterns)
            return [re.compile(pattern) for pattern in patterns]

        elif question_type == 'CQS':
            pattern = self.wildcards_to_python_regex(patterns[0], convert_number_pattern=True)
            return re.compile(pattern)

        else:
            raise ValueError("Question type {} not recognised or supported".format(question_type))

    @staticmethod
    def wildcards_to_python_regex(question, convert_number_pattern=False):
        """Converts wildcard based HTK questions into python's regular expressions.

        Source: https://github.com/CSTR-Edinburgh/merlin/blob/master/src/frontend/label_normalisation.py#L883-L910

        If any wildcards ('*') are present then check if the pattern is at the start or end of the sequence and add
        regex identifiers as necessary ('\A' or '\Z').

        Escape (`re.escape`) non-ASCII and non-numeric characters to ensure no special characters used as a delimiter
        have unwanted effects.

        After escaping, replace HTK wildcards ('*', now '\\*') with python regex characters ('.*'). If a number pattern
        is being searched for, also replace the digit regexes (now '\\(\\\\d\\+\\)') with python regexes ('(\d+)').

        Args:
            question (str): Wildcard-based HTK question regex string.
            convert_number_pattern (bool): If True, then ensure patterns for extracting digit groups are un-escaped.

        Returns:
            (str): string representing valid python regex.
        """
        prefix = ''
        postfix = ''

        # In the case where some wildcards are used, check if we are at the start/end of the sequence.
        if '*' in question:
            # If the regex is at the beginning of the sequence.
            if not question.startswith('*'):
                prefix = '\A'
            # If the regex is at the end of the sequence.
            if not question.endswith('*'):
                postfix = '\Z'

        # Now remove the outer wildcards, these won't be needed as we are using `re.search`.
        question = question.strip('*')

        # Escape the character sequence to ensure no delimiters have unwanted effects.
        question = re.escape(question)

        # Convert remaining HTK wildcards '*' and '?' (now '\\*' and '\\?' due to escape) to python regexes.
        question = question.replace('\\*', '.*')
        question = question.replace('\\?', '.')

        # If we are using a number pattern, we must fix the escaping performed above.
        if convert_number_pattern:
            # integers.
            question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
            # integers or floats.
            question = question.replace('\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')

        return prefix + question + postfix


class SubphoneFeatureSet(object):
    """Container for subphone counter feature extraction.

    Attributes:
        subphone_feature_type (SubphoneFeatureTypeEnum): The type of counter feature set to extract.
    """
    def __init__(self, subphone_feature_type='none'):
        if subphone_feature_type is None:
            subphone_feature_type = 'none'

        self.subphone_feature_type = SubphoneFeatureTypeEnum[subphone_feature_type.upper()]

    @property
    def dim(self):
        if self.subphone_feature_type == SubphoneFeatureTypeEnum.FULL:
            # Zhizheng's original 5 state features + 4 phoneme features.
            return 9
        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.MINIMAL_PHONEME:
            # Equivalent to a frame-based system with minimal features.
            return 3
        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.MINIMAL_FRAME:
            # Minimal features necessary to go from a state-level to frame-level model.
            return 2
        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.FRAME_ONLY:
            # Equivalent to a frame-based system without relying on state-features.
            return 1
        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.STATE_ONLY:
            # Equivalent to a state-based system.
            return 1
        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.UNIFORM_STATE:
            # Equivalent to a frame-based system with uniform state-features.
            return 2
        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.COARSE_CODING:
            # Equivalent to a frame-based positioning system reported in Heiga Zen's work.
            return 2
        else:
            return 0

    @staticmethod
    def relative_pos(index, length):
        """Calculates the relative position of the index through a sequence.

        For consistency with Merlin's subphone features, we do not include fractions starting at 0.0

        Args:
            index (float): The items index starting from 0.
            length (float): The total number of items in the sequence.

        Returns:
            (float): The relative position of `index` through the sequence, forwards. Start = (1/length), End = 1,
            (float): The relative position of `index` through the sequence, backwards. Start = 1, End = (1/length).
        """
        fw = (index + 1.) / length
        bw = (length - index) / length
        return fw, bw

    def full(self, frame_in_state_index, frame_in_phone_index, state_in_phone_index,
             frames_in_state, frames_in_phone, states_per_phone):
        """Zhizheng's original 5 state features + 4 phoneme features.

        NOTE: Only valid for state-level label alignments."""
        fraction_of_state_in_phone = frames_in_state / frames_in_phone

        state_in_phone_index_fw = state_in_phone_index + 1.
        state_in_phone_index_bw = states_per_phone - state_in_phone_index

        fraction_through_state_fw, fraction_through_state_bw = self.relative_pos(frame_in_state_index, frames_in_state)
        fraction_through_phone_fw, fraction_through_phone_bw = self.relative_pos(frame_in_phone_index, frames_in_phone)

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_state_fw, fraction_through_state_bw, frames_in_state,
                state_in_phone_index_fw, state_in_phone_index_bw, frames_in_phone,
                fraction_of_state_in_phone, fraction_through_phone_bw, fraction_through_phone_fw]

    def minimal_phoneme(self, frame_in_phone_index, frames_in_phone):
        """Equivalent to a frame-based system with minimal features."""
        fraction_through_phone_fw, fraction_through_phone_bw = self.relative_pos(frame_in_phone_index, frames_in_phone)

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_phone_fw, fraction_through_phone_bw, frame_in_phone_index]

    def minimal_frame(self, frame_in_state_index, state_in_phone_index, frames_in_state):
        """Minimal features necessary to go from a state-level to frame-level model."""
        fraction_through_state_fw, _ = self.relative_pos(frame_in_state_index, frames_in_state)

        state_in_phone_index_fw = state_in_phone_index + 1.

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_state_fw, state_in_phone_index_fw]

    def frame_only(self, frame_in_phone_index, frames_in_phone):
        """Equivalent to a frame-based system without relying on state-features."""
        fraction_through_phone_fw, _ = self.relative_pos(frame_in_phone_index, frames_in_phone)

        return [fraction_through_phone_fw]

    def state_only(self, state_in_phone_index):
        """Equivalent to a state-based system.

        NOTE: Only valid for state-level labels alignments."""
        state_in_phone_index_fw = state_in_phone_index + 1.
        return [state_in_phone_index_fw]

    def uniform_state(self, frame_in_phone_index, frames_in_phone, states_per_phone):
        """Equivalent to a frame-based system with uniform state-features."""
        fraction_through_phone_fw, _ = self.relative_pos(frame_in_phone_index, frames_in_phone)

        # Ignore state_index and assume state durations are uniform
        uniform_state_index = np.ceil(frame_in_phone_index / frames_in_phone * states_per_phone)

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_phone_fw, uniform_state_index]

    def coarse_coding(self, frame_in_phone_index, frames_in_phone):
        """Equivalent to a frame-based positioning system reported in Heiga Zen's work."""
        offset = frame_in_phone_index / frames_in_phone  # [0.0, 1.0]

        x1_in_normal_pdf = -1.0 + offset  # [-1.0, 0.0]
        x2_in_normal_pdf = -0.5 + offset  # [ 0.5, 0.5]
        x3_in_normal_pdf =  0.0 + offset  # [ 0.0, 1.0]

        mu, sigma = 0.0, 0.4
        densities_from_normal_pdf = norm(mu, sigma).pdf([x1_in_normal_pdf, x2_in_normal_pdf, x3_in_normal_pdf])

        return densities_from_normal_pdf + [frames_in_phone]

    def query(self, frame_in_state_index, frame_in_phone_index, state_in_phone_index,
              frames_in_state, frames_in_phone, states_per_phone):
        """Creates the subphone counter features based on the stateful position information of the subphone.

        Args:
            frame_in_state_index (int): Index of this frame through the current state.
            frame_in_phone_index (int): Index of this frame through the current phone.
            state_in_phone_index (int): Index of this state through the current phone, range = [0, states_per_phone-1].
            frames_in_state (int): Number of frames in the current state.
            frames_in_phone (int): Number of frames in the current phone.
            states_per_phone (int): Number of states in each phone, only valid for state-level label alignments.

        Returns:
            (np.ndarray): Subphone counter features for one frame.
        """
        frame_in_state_index = float(frame_in_state_index)
        frame_in_phone_index = float(frame_in_phone_index)
        state_in_phone_index = float(state_in_phone_index)
        frames_in_state = float(frames_in_state)
        frames_in_phone = float(frames_in_phone)
        states_per_phone = float(states_per_phone)

        if self.subphone_feature_type == SubphoneFeatureTypeEnum.FULL:
            # Zhizheng's original 5 state features + 4 phoneme features.
            return self.full(frame_in_state_index, frame_in_phone_index, state_in_phone_index,
                             frames_in_state, frames_in_phone, states_per_phone)

        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.MINIMAL_PHONEME:
            # Equivalent to a frame-based system with minimal features.
            return self.minimal_phoneme(frame_in_phone_index, frames_in_phone)

        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.MINIMAL_FRAME:
            # Minimal features necessary to go from a state-level to frame-level model.
            return self.minimal_frame(frame_in_state_index, state_in_phone_index, frames_in_state)

        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.FRAME_ONLY:
            # Equivalent to a frame-based system without relying on state-features.
            return self.frame_only(frame_in_phone_index, frames_in_phone)

        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.STATE_ONLY:
            # Equivalent to a state-based system.
            return self.state_only(state_in_phone_index)

        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.UNIFORM_STATE:
            # Equivalent to a frame-based system with uniform state-features.
            return self.uniform_state(frame_in_phone_index, frames_in_phone, states_per_phone)

        elif self.subphone_feature_type == SubphoneFeatureTypeEnum.COARSE_CODING:
            # Equivalent to a frame-based positioning system reported in Heiga Zen's work.
            return self.coarse_coding(frame_in_phone_index, frames_in_phone)

        else:
            return []


class Label(object):
    """Container for full-context labels, allows for binarising of labels.

    Attributes:
        file_path (str): Label file to be loaded.
        state_level (bool): If True, the labels should be duplicated `self.states_per_phone` times per phone.
        states_per_phone (int): Number of states in a phone. If `self.state_level` is false, then this will be 1.
        lines (list<str>): Lines from the label file, containing state information if present.
        labels (list<str>): The de-duplicated labels with no start/end times.
        phones (list<str>): The phone identities.
        state_in_phone_durations (list<list<int>>): Inner list indicates the number of frames for the states in a phone,
            If `self.state_level` is false then each inner list will be a singleton list, equalling the phone duration.
        phone_durations (list<int>): Each item indicates the number of frames in a phone.
    """
    def __init__(self, file_path, state_level=True, states_per_phone=STATES_PER_PHONE):
        """Loads the label from `file_path` and processes basic information, preparing it for querying.

        Args:
            file_path (str): Label file to be loaded.
            state_level (bool): If True, the labels should be duplicated `self.states_per_phone` times per phone.
            states_per_phone (int): Number of states in a phone. If `self.state_level` is false, then this will be 1.
        """
        self.file_path = file_path
        self.base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.state_level = state_level
        self.states_per_phone = states_per_phone if state_level else 1

        self.lines = file_io.load_lines(self.file_path)
        # Ensure the all whitespaces are single space characters.
        self.lines = list(map(lambda l: re.sub('\s+', ' ', l), self.lines))

        # Extracted labels will not be duplicated for each state in phone.
        self.labels = self.trim_labels(self.state_level)
        self.phones = self.extract_phone_identities()

        # If `self.state_level` is false, then each item in `self.state_in_phone_durations` will be a singleton list.
        self.state_in_phone_durations, self.phone_durations = self.extract_durations()

    def __repr__(self):
        return '\n'.join(self.lines)

    def __str__(self):
        return ' '.join(self.phones)

    def trim_labels(self, state_level):
        """Removes start and end times (if present) and state information. Removes duplicated state-level labels.

        Args:
            state_level (bool): If True, remove state information and duplicate labels.

        Returns:
            (list<str>): Non-duplicated full context labels without timing or state information."""
        labels = []
        for i in range(0, len(self.lines), self.states_per_phone):
            line = self.lines[i]

            # Remove the space-delimited start and end times if they are present. If they are not present, taking the
            # last element after splitting on spaces is an identity operation as there will be no spaces present.
            label = line.split(' ')[-1]

            # Remove the state indicator '[i]' if it exists.
            if state_level:
                label = label[:-3]

            labels.append(label)

        return labels

    def extract_phone_identities(self):
        """Searches for the phone identity in each label.

        Returns:
            (list<str>): List of phone identities."""
        current_phone_regex = re.compile('-(.+?)\+')

        phones = []
        for label in self.labels:
            current_phone_match = current_phone_regex.search(label)
            current_phone = current_phone_match.group(1)

            phones.append(current_phone)

        return phones

    def extract_durations(self):
        """Counts the number of frames for each phone, i.e. the duration of each phone.

        Returns:
            (np.ndarray[n_phones, n_states]): Integer durations per state in phone, `states_per_phone` values per phone.
            (np.ndarray[n_phones]): Integer durations per phone."""
        durations = []
        for i in range(0, len(self.lines), self.states_per_phone):
            if self.state_level:
                lines = self.lines[i:i+self.states_per_phone]
            else:
                lines = [self.lines[i]]

            phone_duration = []
            for line in lines:
                start, end, _ = line.split(' ')

                frame_index_start = int(start) / (FRAME_SHIFT_MS * 10000)
                frame_index_end = int(end) / (FRAME_SHIFT_MS * 10000)

                phone_duration.append(frame_index_end - frame_index_start)

            durations.append(phone_duration)

        state_level_durations = np.array(durations, dtype=np.int32)
        phone_level_durations = np.sum(state_level_durations, axis=1, dtype=np.int32)

        return state_level_durations, phone_level_durations

    def extract_counter_features(self, subphone_feature_set):
        """Creates the frame-level counter features.

        Args:
            subphone_feature_set (SubphoneFeatureSet instance): Container that defines the subphone features to be
                extracted from the durations. If None, then no additional frame-level features are created.

        Returns:
            (np.ndarray): Numerical counter features at the frame level."""
        n_frames = np.sum(self.phone_durations).item()
        counter_dim = subphone_feature_set.dim
        counter_features= np.zeros((n_frames, counter_dim), dtype=np.float32)

        frame_index = 0
        for label, state_in_phone_duration in zip(self.labels, self.state_in_phone_durations):
            frames_in_phone = sum(state_in_phone_duration)

            # Track the frame counter per phone, so we don't reset it after each iteration of the inner loop.
            frame_in_phone_index = 0
            for state_in_phone_index, frames_in_state in enumerate(state_in_phone_duration):

                # We can't track `frame_in_phone_index` here as it would reset for each state.
                for frame_in_state_index in range(frames_in_state):

                    # Get the subphone counter features for this frame.
                    counter_feature = subphone_feature_set.query(
                        frame_in_state_index, frame_in_phone_index, state_in_phone_index,
                        frames_in_state, frames_in_phone, self.states_per_phone)

                    # Add the frame-level counters to our accumulator array.
                    counter_features[frame_index] = counter_feature
                    frame_in_phone_index += 1
                    frame_index += 1

        print("Numerical labels created for {}: {} frames; {} subphone counter features."
              .format(self.base_name, n_frames, counter_dim))

        return counter_features

    def extract_numerical_labels(self, question_set, upsample_to_frame_level=True):
        """Queries the labels using the question set, and returns the numerical labels.

        Args:
            question_set (QuestionSet instance): Question set used to query the labels.
            upsample_to_frame_level (bool): If True, upsamples phone-level features to frame-level.

        Returns:
            (np.ndarray): Numerical labels suitable for machine learning."""
        if upsample_to_frame_level:
            seq_len = np.sum(self.phone_durations).item()
        else:
            seq_len = len(self.phones)
        lab_dim = question_set.dim

        numerical_labels = np.zeros((seq_len, lab_dim), dtype=np.float32)

        if upsample_to_frame_level:
            frame_index = 0
            for label, state_in_phone_duration in zip(self.labels, self.state_in_phone_durations):
                # Get the numerical label once for each phone using the question set.
                label_vector = question_set.query(label)

                for frames_in_state in state_in_phone_duration:
                    for _ in range(frames_in_state):
                        # Add the phone-level label to the frame-level accumulator array.
                        numerical_labels[frame_index] = label_vector
                        frame_index += 1

            print("Numerical labels created for {}, upsampled to frame-level: {} phones; {} frames; {} label features."
                  .format(self.base_name, len(self.phones), seq_len, lab_dim))

        else:
            for phone_index, label in enumerate(self.labels):
                # Get the numerical label once for each phone using the question set.
                label_vector = question_set.query(label)
                numerical_labels[phone_index] = label_vector

            print("Numerical labels created for {}, at phone-level: {} phones; {} label features."
                  .format(self.base_name, seq_len, lab_dim))

        return numerical_labels


def process(lab_dir, id_list, out_dir, state_level,
            question_file, upsample, subphone_feat_type, calculate_normalisation):
    """Processes label files in id_list, saves the numerical labels and durations to file.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
        question_file (str): Question set to be loaded. Can be one of the four provided question sets;
                questions-unilex_dnn_600.hed
                questions-unilex_phones_69.hed
                questions-radio_dnn_416.hed
                questions-radio_phones_48.hed
                questions-mandarin.hed
                questions-japanese.hed
        upsample (bool): Whether to upsample phone-level numerical labels to frame-level.
        subphone_feat_type (str): Subphone features to be extracted from the durations.
        calculate_normalisation (bool): Calculate mean-variance and min-max normalisation for duration and labels.
    """
    file_ids = get_file_ids(lab_dir, id_list)
    question_set = QuestionSet(question_file)
    subphone_feature_set = SubphoneFeatureSet(subphone_feat_type)

    for file_id in file_ids:
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = Label(lab_path, state_level)

        numerical_labels = label.extract_numerical_labels(question_set, upsample_to_frame_level=upsample)
        counter_features = label.extract_counter_features(subphone_feature_set)
        durations = label.phone_durations.reshape((-1, 1))
        phones = label.phones

        n_frames = np.sum(durations).item()
        n_phones = len(label.phones)

        file_io.save_bin(numerical_labels, os.path.join(out_dir, 'lab', '{}.lab'.format(file_id)))
        file_io.save_bin(counter_features, os.path.join(out_dir, 'counters', '{}.counters'.format(file_id)))
        file_io.save_txt(durations, os.path.join(out_dir, 'dur', '{}.dur'.format(file_id)))
        file_io.save_lines(phones, os.path.join(out_dir, 'phones', '{}.txt'.format(file_id)))

        file_io.save_txt(n_frames, os.path.join(out_dir, 'n_frames', '{}.txt'.format(file_id)))
        file_io.save_txt(n_phones, os.path.join(out_dir, 'n_phones', '{}.txt'.format(file_id)))

    if calculate_normalisation:
        process_minmax(out_dir, 'lab', id_list)
        process_minmax(out_dir, 'counters', id_list)
        process_mvn(out_dir, 'dur', is_npy=False, id_list=id_list, deltas=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extracts numerical labels, counter features, and durations from forced alignment labels files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.id_list, args.out_dir, args.state_level,
            args.question_file, args.upsample_to_frame_level, args.subphone_feat_type, args.calculate_normalisation)


if __name__ == "__main__":
    main()

