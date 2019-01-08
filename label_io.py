"""Handles loading and modifying label files.

Usage:
    python label_io.py --lab_file FILE"""

import argparse
import re

import numpy as np
from scipy.stats import norm

from file_io import save_bin

STATES_PER_PHONE = 5
FRAME_SHIFT_MS = 5


def add_arguments(parser):
    parser.add_argument("--state_level", "-s", action="store_true", dest="state_level", default=True,
                        help="Is the label file state level (or frame level).")
    parser.add_argument("--question_file", action="store", dest="question_file", type=str, required=True,
                        help="File containing the '.hed' question set to query the labels with.")
    parser.add_argument("--subphone_feat_type", action="store", dest="subphone_feat_type", type=str, default=None,
                        help="The type of subphone counter features to add to the frame-level numerical vectors.")


def load_txt(file_path):
    """Loads text data from a text file.

    Args:
        file_path (str): File to load the text from.

    Returns:
        (list<str>) Sequence of strings."""
    with open(file_path, 'r') as f:
        lines = list(map(str.strip, f.readlines()))

    return lines


def save_txt(lines, file_path):
    """Saves text in a text file.

    Args:
        lines (list<str>): Sequence of strings.
        file_path (str): File to save the text to."""
    lines = list(map(lambda x: '{}\n'.format(line) for line in lines))

    with open(file_path, 'w') as f:
        f.writelines(lines)


def hed_to_new_format(hed_file_name):
    """Converts the question set '.hed' format to new format based on LABEL_DESCRIPTION keys.

    '.hed' format:
    QS  "C-Voiced_Plosive"  {*-b+*,*-d+*,*-g+*}
    QS  "RR-ou1"            {*=ou1:*}
    QS  "C-Syl_oo1"         {*|oo1/C/*}
    CQS "Seg_Fw"            {:(\d+)_}

    new format:
    "C-Voiced_Plosive"  p3  IN      - [b,d,g]   +
    "RR-ou1"            p5  IN      = [ou1]     @
    "C-Syl_oo1"         b16 IN      | [oo1]     /C/
    "Seg_Fw"            p6  VALUE   @ []        _
    """
    pass


class QuestionSet(object):
    """Container for qustion set regex questions."""
    def __init__(self, file_path):
        """...

        :param file_path:
        """
        self.file_path = file_path
        self.lines = load_txt(self.file_path)
        # Ensure the only whitespaces are single space characters.
        self.lines = list(map(lambda l: re.sub('\s+', ' ', l), self.lines))

        self.binary_regexes, self.numerical_regexes = self.compile_questions(self.lines)


    def query(self, string):
        binary_features = self.binary_queries(string)
        numerical_features = self.numerical_queries(string)
        return np.concatenate((binary_features, numerical_features))

    def binary_queries(self, string):
        vector = np.zeros(len(self.binary_regexes), dtype=np.float32)
        for i, (name, patterns) in enumerate(self.binary_regexes):
            for pattern in patterns:
                # If any of the patterns evaluate to true then slip the bit and move to the next set of patterns.
                if pattern.search(string):
                    vector[i] = 1
                    break
        return vector

    def numerical_queries(self, string):
        vector = np.zeros(len(self.numerical_regexes), dtype=np.float32)
        for i, (name, pattern) in enumerate(self.numerical_regexes):
            match = pattern.search(string)
            if match:
                vector[i] = match.group(1)
            else:
                vector[i] = -1.
        return vector

    def compile_questions(self, lines):
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
    """Container for subphone counter feature extraction."""
    def __init__(self, subphone_feature_type):
        supported_subphone_feature_type = [
            'full', 'minimal_phoneme', 'minimal_frame', 'frame_only', 'state_only', 'uniform_state', 'coarse_coding']
        if subphone_feature_type not in supported_subphone_feature_type:
            raise TypeError("Unknown/unsupported value for subphone_feature_type: {}".format(subphone_feature_type))

        self.subphone_feature_type = subphone_feature_type

    @staticmethod
    def relative_pos(index, length):
        """Calcaulates the relative position of the index through a sequence.

        For consistency with Merlin's subphone features, we do not include fractions starting at 0.0

        Args:
            index (float): The items index starting from 0.
            length (float): The total number of items in the sequence.

        Returns:
            (float): The relative position of `index` through the sequence, forwards. Start = (1/length), End = 1
            (float): The relative position of `index` through the sequence, backwards. Start = 1, End = (1/length)
        """
        fw = (index + 1.) / length
        bw = (length - index) / length
        return fw, bw

    def full(self, frame_index, frame_index_in_phone, state_index, frames_in_state, frames_in_phone):
        """Zhizheng's original 5 state features + 4 phoneme features.

        NOTE: Only valid for state-level labels alignments."""
        fraction_of_state_in_phone = frames_in_state / frames_in_phone

        state_index_fw = state_index + 1.
        state_index_bw = float(STATES_PER_PHONE) - state_index

        fraction_through_state_fw, fraction_through_state_bw = self.relative_pos(frame_index, frames_in_state)
        fraction_through_phone_fw, fraction_through_phone_bw = self.relative_pos(frame_index_in_phone, frames_in_phone)

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_state_fw, fraction_through_state_bw, frames_in_state, state_index_fw, state_index_bw,
                frames_in_phone, fraction_of_state_in_phone, fraction_through_phone_bw, fraction_through_phone_fw]

    def minimal_phoneme(self, frame_index_in_phone, frames_in_phone):
        """Equivalent to a frame-based system with minimal features."""
        fraction_through_phone_fw, fraction_through_phone_bw = self.relative_pos(frame_index_in_phone, frames_in_phone)

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_phone_fw, fraction_through_phone_bw, frame_index_in_phone]

    def minimal_frame(self, frame_index, state_index, frames_in_state):
        """Minimal features necessary to go from a state-level to frame-level model."""
        fraction_through_state_fw, _ = self.relative_pos(frame_index, frames_in_state)

        state_index_fw = state_index + 1.

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_state_fw, state_index_fw]

    def frame_only(self, frame_index_in_phone, frames_in_phone):
        """Equivalent to a frame-based system without relying on state-features."""
        fraction_through_phone_fw, _ = self.relative_pos(frame_index_in_phone, frames_in_phone)

        return [fraction_through_phone_fw]

    def state_only(self, state_index):
        """Equivalent to a state-based system.

        NOTE: Only valid for state-level labels alignments."""
        state_index_fw = state_index + 1.
        return [state_index_fw]

    def uniform_state(self, frame_index_in_phone, frames_in_phone):
        """Equivalent to a frame-based system with uniform state-features."""
        fraction_through_phone_fw, _ = self.relative_pos(frame_index_in_phone, frames_in_phone)

        # Ignore state_index and assume state durations are uniform
        uniform_state_index = np.ceil(frame_index_in_phone / frames_in_phone * STATES_PER_PHONE)

        # The order of the features is determined based on Merlin's subphone features.
        return [fraction_through_phone_fw, uniform_state_index]

    def coarse_coding(self, frame_index_in_phone, frames_in_phone):
        """Equivalent to a frame-based positioning system reported in Heiga Zen's work."""
        offset = frame_index_in_phone / frames_in_phone  # [0.0, 1.0]

        x1_in_normal_pdf = -1.0 + offset  # [-1.0, 0.0]
        x2_in_normal_pdf = -0.5 + offset  # [ 0.5, 0.5]
        x3_in_normal_pdf =  0.0 + offset  # [ 0.0, 1.0]

        mu, sigma = 0.0, 0.4
        densities_from_normal_pdf = norm(mu, sigma).pdf([x1_in_normal_pdf, x2_in_normal_pdf, x3_in_normal_pdf])

        return [*densities_from_normal_pdf, frames_in_phone]

    def query(self, frame_index, frame_index_in_phone, state_index, frames_in_state, frames_in_phone):
        """...

        Args:
            frame_index (int): The index of this frame through the current state.
            frame_index_in_phone (int): The index of this frame through the current phone.
            state_index (int): The index of this state through the current phone, range = [1, STATES_PER_PHONE].
            frames_in_state (int): The number of frames in the current state.
            frames_in_phone (int): The number of frames in the current phone.

        Returns:
            (np.ndarray): Subphone counter features for one frame.
        """
        frame_index = float(frame_index)
        frame_index_in_phone = float(frame_index_in_phone)
        state_index = float(state_index)
        frames_in_state = float(frames_in_state)
        frames_in_phone = float(frames_in_phone)

        if self.subphone_feature_type == 'full':
            # Zhizheng's original 5 state features + 4 phoneme features.
            return self.full(frame_index, frame_index_in_phone, state_index, frames_in_state, frames_in_phone)

        elif self.subphone_feature_type == 'minimal_phoneme':
            # Equivalent to a frame-based system with minimal features.
            return self.minimal_phoneme(frame_index_in_phone, frames_in_phone)

        elif self.subphone_feature_type == 'minimal_frame':
            # Minimal features necessary to go from a state-level to frame-level model.
            return self.minimal_frame(frame_index, state_index, frames_in_state)

        elif self.subphone_feature_type == 'frame_only':
            # Equivalent to a frame-based system without relying on state-features.
            return self.frame_only(frame_index_in_phone, frames_in_phone)

        elif self.subphone_feature_type == 'state_only':
            # Equivalent to a state-based system.
            return self.state_only(state_index)

        elif self.subphone_feature_type == 'uniform_state':
            # Equivalent to a frame-based system with uniform state-features.
            return self.uniform_state(frame_index_in_phone, frames_in_phone)

        elif self.subphone_feature_type == 'coarse_coding':
            # Equivalent to a frame-based positioning system reported in Heiga Zen's work.
            return self.coarse_coding(frame_index_in_phone, frames_in_phone)


class Label(object):
    """Container for full-context labels, allows for binarising of labels."""
    def __init__(self, file_path, state_level=True):
        """...

        Args:
            file_path (str):
            state_level (bool):
        """
        self.file_path = file_path
        self.state_level = state_level

        self.lines = load_txt(self.file_path)
        # Ensure the only whitespaces are single space characters.
        self.lines = list(map(lambda l: re.sub('\s+', ' ', l), self.lines))

        self.labels = self.trim_labels(self.state_level)

        # If `state_level == False` then each item in `self.state_in_phone_durations` will be a singleton list.
        self.state_in_phone_durations, self.phone_durations = self.extract_durations(self.state_level)

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

        states_per_phone = STATES_PER_PHONE if state_level else 1
        for i in range(0, len(self.lines), states_per_phone):
            line = self.lines[i]

            # Remove the space-delimited start and end times if they are present. If they are not present, taking the
            # last element after splitting on spaces is an identity operation as there will be no spaces present.
            label = line.split(' ')[-1]

            # Remove the state indicator '[i]' if it exists.
            if state_level:
                label = label[:-3]

            labels.append(label)

        return labels

    def extract_durations(self, state_level):
        """Counts the number of frames for each phone, i.e. the duration of each phone.

        Returns:
            (np.ndarray[n_phones, n_states]): Integer durations per state in phone, `states_per_phone` values per phone.
            (np.ndarray[n_phones]): Integer durations per phone."""
        durations = []

        states_per_phone = STATES_PER_PHONE if state_level else 1
        for i in range(0, len(self.lines), states_per_phone):
            if state_level:
                lines = self.lines[i:i+STATES_PER_PHONE]
            else:
                lines = [self.lines[i]]

            phone_duration = []
            for line in lines:
                start, end, label = line.split(' ')

                frame_index_start = int(start) / (FRAME_SHIFT_MS * 10000)
                frame_index_end = int(end) / (FRAME_SHIFT_MS * 10000)

                phone_duration.append(frame_index_end - frame_index_start)

            durations.append(phone_duration)

        state_level_durations = np.array(durations, dtype=np.int32)
        phone_level_durations = np.sum(state_level_durations, axis=1, dtype=np.int32)

        return state_level_durations, phone_level_durations

    def binarise(self, question_set, subphone_feature_set=None, upsample_to_frame_level=True):
        """Queries the labels using the question set, calculates any additional features, and returns vectorised labels.

        Args:
            question_set (QuestionSet instance): Question set used to query the labels.
            subphone_feature_set (SubphoneFeatureSet instance): Container that defines the subphone features to be
                extracted from the durations. If None, then no additional frame-level features are added.
            upsample_to_frame_level (bool): If True, upsamples phone-level features to frame-level, and subphone counter
                features are added per frame using `subphone_feature_set`. If False, `subphone_feature_set` is ignored.

        Returns:
            (np.ndarray): Numerical labels suitable for machine learning."""
        # Accumulator array used to add frame-level vectors to.
        frame_level_vectors = []

        for label, phone_duration in zip(self.labels, self.state_in_phone_durations):
            # Get the numerical label once for each phone using the question set.
            label_vector = question_set.query(label)

            if upsample_to_frame_level:
                frames_in_phone = sum(phone_duration)

                # Track the frame counter per phone, so we don't reset it after each iteration of the inner loop.
                frame_index_in_phone = 0
                for state_index, frames_in_state in enumerate(phone_duration):
                    # We can't track `frame_index` here as it would reset for each state.
                    for frame_index in range(frames_in_state):
                        # Get the subphone counter features for this frame.
                        if subphone_feature_set:
                            subphone_features = subphone_feature_set.query(
                                frame_index, frame_index_in_phone, state_index, frames_in_state, frames_in_phone)
                        else:
                            subphone_features = []

                        # Add the phone-level label and the frame-level counters to our accumulator array.
                        frame_level_vectors.append(np.concatenate((label_vector, subphone_features)))
                        frame_index_in_phone += 1
            else:
                frame_level_vectors.append(label_vector)

        number_of_frames = len(frame_level_vectors)
        question_set_dim = np.array(label_vector).shape[0]
        suphone_feature_dim = np.array(subphone_features).shape[0]
        total_dimensionality = np.array(frame_level_vectors[0]).shape[0]
        print("Numerical labels created: {} frames; {} question features; {} subphone counter features; and {} total "
              "features.".format(number_of_frames, question_set_dim, suphone_feature_dim, total_dimensionality))

        return np.array(frame_level_vectors, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to load label files.")
    parser.add_argument("--lab_file", action="store", dest="lab_file", type=str, required=True,
                        help="File path of the label to be converted.")
    parser.add_argument("--out_file", action="store", dest="out_file", type=str, required=True,
                        help="File path to save the numerical labels to.")
    add_arguments(parser)
    args = parser.parse_args()

    label = Label(args.lab_file, args.state_level)
    questions = QuestionSet(args.question_file)
    if args.subphone_feat_type:
        suphone_features = SubphoneFeatureSet(args.subphone_feat_type)
    else:
        suphone_features = None

    numerical_labels = label.binarise(questions, suphone_features)
    save_bin(numerical_labels, args.out_file)

