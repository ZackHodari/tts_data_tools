"""Handles loading and modifying label files.

Usage:
    python label_io.py --lab_file FILE"""

import argparse
import re

import numpy as np

LABEL_DESCRIPTION = {
    't1': "the start time of the current phone in 1/10,000th of a millisecond",
    't2': "the end time of the current phone in 1/10,000th of a millisecond",
    'p1': "the phoneme identity before the previous phoneme",
    'p2': "the previous phoneme identity",
    'p3': "the current phoneme identity",
    'p4': "the next phoneme identity",
    'p5': "the phoneme after the next phoneme identity",
    'p6': "position of the current phoneme identity in the current syllable (forward)",
    'p7': "position of the current phoneme identity in the current syllable (backward)",
    'a1': "whether the previous syllable stressed or not (0: not stressed, 1: stressed)",
    'a2': "whether the previous syllable accented or not (0: not accented, 1: accented)",
    'a3': "the number of phonemes in the previous syllable",
    'b1': "whether the current syllable stressed or not (0: not stressed, 1: stressed)",
    'b2': "whether the current syllable accented or not (0: not accented, 1: accented)",
    'b3': "the number of phonemes in the current syllable",
    'b4': "position of the current syllable in the current word (forward)",
    'b5': "position of the current syllable in the current word (backward)",
    'b6': "position of the current syllable in the current phrase (forward)",
    'b7': "position of the current syllable in the current phrase (backward)",
    'b8': "the number of stressed syllables before the current syllable in the current phrase",
    'b9': "the number of stressed syllables after the current syllable in the current phrase",
    'b10': "the number of accented syllables before the current syllable in the current phrase",
    'b11': "the number of accented syllables after the current syllable in the current phrase",
    'b12': "the number of syllables from the previous stressed syllable to the current syllable",
    'b13': "the number of syllables from the current syllable to the next stressed syllable",
    'b14': "the number of syllables from the previous accented syllable to the current syllable",
    'b15': "the number of syllables from the current syllable to the next accented syllable",
    'b16': "name of the vowel of the current syllable",
    'c1': "whether the next syllable stressed or not (0: not stressed, 1: stressed)",
    'c2': "whether the next syllable accented or not (0: not accented, 1: accented)",
    'c3': "the number of phonemes in the next syllable",
    'd1': "gpos (guess part-of-speech) of the previous word",
    'd2': "the number of syllables in the previous word",
    'e1': "gpos (guess part-of-speech) of the current word",
    'e2': "the number of syllables in the current word",
    'e3': "position of the current word in the current phrase (forward)",
    'e4': "position of the current word in the current phrase (backward)",
    'e5': "the number of content words before the current word in the current phrase",
    'e6': "the number of content words after the current word in the current phrase",
    'e7': "the number of words from the previous content word to the current word",
    'e8': "the number of words from the current word to the next content word",
    'f1': "gpos (guess part-of-speech) of the next word",
    'f2': "the number of syllables in the next word",
    'g1': "the number of syllables in the previous phrase",
    'g2': "the number of words in the previous phrase",
    'h1': "the number of syllables in the current phrase",
    'h2': "the number of words in the current phrase",
    'h3': "position of the current phrase in utterence (forward)",
    'h4': "position of the current phrase in utterence (backward)",
    'h5': "TOBI endtone of the current phrase",
    'i1': "the number of syllables in the next phrase",
    'i2': "the number of words in the next phrase",
    'j1': "the number of syllables in this utterence",
    'j2': "the number of words in this utternce",
    'j3': "the number of phrases in this utterence"
}


def add_arguments(parser):
    parser.add_argument("--state_level", "-s", action="store_true", dest="state_level", default=True,
                        help="Is the label file state level (or frame level).")


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


"""
1100000 1200000xx~#-b+ei=s:1_4/A/0_0_0/B/1-1-4:1-1&1-9#1-6$1-4>0-1<0-5|ei/C/1+0+2/D/0_0/E/content+1:1+7&1+3#0+3/F/in_1/G/0_0/H/9=7:1=1&L-L%/I/0_0/J/9+7-1[6]

1100000 1200000
xx~#-b+ei=s:1_4
/A/ 0_0_0
/B/ 1-1-4 :1-1 &1-9 #1-6 $1-4 >0-1 <0-5 |ei
/C/ 1+0+2
/D/ 0_0
/E/ content+1 :1+7 &1+3 #0+3
/F/ in_1
/G/ 0_0
/H/ 9=7 :1=1 &L-L%
/I/ 0_0
/J/ 9+7-1
[6]

1100000
1200000

xx~# -b +ei=s  :1 _4
p1ˆp2-p3+p4=p5 @p6_p7

/A/ 0 _0 _0
/A: a1_a2_a3

/B/ 1 -1 -4  :1 -1  &1 -9  #1 -6  $1 -4    >0  -1   <0   -5  |ei
/B: b1-b2-b3 @b4-b5 &b6-b7 #b8-b9 $b10-b11 !b12-b13 ;b14-b15 |b16

/C/ 1 +0 +2
/C: c1+c2+c3

/D/ 0 _0
/D: d1_d2

/E/ content +1  :1  +7  &1  +3  #0  +3
/E: e1      +e2 @e3 +e4 &e5 +e6 #e7 +e8

/F/ in_1
/F: f1_f2

/G/ 0 _0
/G: g1_g2

/H/ 9 =7  :1 =1  &L-L%
/H: h1=h2 @h3=h4 |h5

/I/ 0 _0
/I: i1_i2

/J/ 9  +7  -1
/J: j1 +j2 -j3

[6]
"""


class QuestionSet(object):
    """Container for qustion set regex questions."""
    def __init__(self, file_path):
        """...

        :param file_path:
        """
        self.file_path = file_path
        self.lines = load_txt(self.file_path)

        self.binary_regexes = []
        self.numerical_regexes = []

    def query(self, string):
        binary_features = self.binary_queries(string)
        numerical_features = self.numerical_queries(string)
        return np.concatenate((binary_features, numerical_features), axis=1)

    def binary_queries(self, string):
        vector = np.zeros(len(self.binary_regexes), dtype=np.float32)
        for i, (name, patterns) in enumerate(self.binary_regexes.items()):
            for pattern in patterns:
                # If any of the patterns evaluate to true then slip the bit and move to the next set of patterns.
                if pattern.search(string):
                    vector[i] = 1
                    break
        return vector

    def numerical_queries(self, string):
        vector = np.zeros(len(self.numerical_regexes), dtype=np.float32)
        for i, (name, pattern) in enumerate(self.numerical_regexes.items()):
            match = pattern.search(string)
            if match:
                vector[i] = match.group(1)
            else:
                vector[i] = -1.
        return vector


    def compile_questions(self):
        for line in self.lines:
            # Consolidate whitespace and separate the space-delimited line.
            line = re.sub('\s+', ' ', line)
            question_type, name, patterns = line.split(' ')

            # Remove the quotes from around the name.
            name = name[1:-1]

            # Remove the brackets around the patterns, and split into a list of patterns.
            patterns = patterns[1:-1].split(',')

            if question_type == 'QS':
                patterns = map(self.wildcards_to_python_regex, patterns)
                regexes = [re.compile(pattern) for pattern in patterns]
                self.binary_regexes.appemd((name, regexes))

            elif question_type == 'CQS':
                pattern = self.wildcards_to_python_regex(patterns[0], convert_number_pattern=True)
                regex = re.compile(pattern)
                self.numerical_regexes.append((name, regex))

            else:
                raise ValueError("Question type {} not recognised or supported".format(question_type))

    def wildcards_to_python_regex(self, question, convert_number_pattern=False):
        """Converts wildcard based HTK questions into python's regular expressions.

        Source: https://github.com/CSTR-Edinburgh/merlin/blob/master/src/frontend/label_normalisation.py#L883-L910

        If any wildcards ('*') are present then check if the pattern is at the start or end of the sequence and add
        regex identifiers as necessary ('\A' or '\Z').

        Escape (`re.escape`) non-ASCII and non-numeric characters to ensure no special characters used as a delimiter
        have unwanted effects.

        After escaping, replace HTK wildcards ('*', now '\\*') with python regex characters ('.*'). If a number pattern
        is being searched for, also replace the digit regexes (now '\\(\\\\d\\+\\)') with python regexes ('(\d+)').

        :param pattern: 
        :return: 
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

    def extract_groups(self, line):
        _regex = '(?P<{}>{pattern})'
        _binary = '(?P<{}>0|1)'
        _number = '(?P<{}>\d+)'
        _phone = '(?P<{}>[@|a-z]{1,3}\d?)'
        _gpos= '(?P<{}>[a-z]*|0)'

        regex = ("({t1} {t2} )?"
                 "{p1}ˆ{p2}-{p3}+{p4}={p5}@{p6}_{p7}"
                 "/A/{a1}_{a2}_{a3}"
                 "/B/{b1}-{b2}-{b3}@{b4}-{b5}&{b6}-{b7}#{b8}-{b9}${b10}-{b11}!{b12}-{b13};{b14}-{b15}|{b16}"
                 "/C/{c1}+{c2}+{c3}"
                 "/D/{d1}_{d2}"
                 "/E/{e1}+{e2}@{e3}+{e4}&{e5}+{e6}#{e7}+{e8}"
                 "/F/{f1}_{f2}"
                 "/G/{g1}_{g2}"
                 "/H/{h1}={h2}@{h3}={h4}|{h5}"
                 "/I/{i1}_{i2}"
                 "/J/{j1}+{j2}-{j3}\[{state}\]".format(
            t1=_number.format('t1'), t2=_number.format('t2'),
            p1=_phone.format('p1'), p2=_phone.format('p2'), p3=_phone.format('p3'), p4=_phone.format('p4'),
            p5=_phone.format('p5'), p6=_number.format('p6'), p7=_number.format('p7'),
            a1=_binary.format('a1'), a2=_binary.format('a2'), a3=_number.format('a3'),
            b1=_binary.format('b1'), b2=_binary.format('b2'), b3=_number.format('b3'), b4=_number.format('b4'),
            b5=_number.format('b5'), b6=_number.format('b6'), b7=_number.format('b7'), b8=_number.format('b8'),
            b9=_number.format('b9'), b10=_number.format('b10'), b11=_number.format('b11'), b12=_number.format('b12'),
            b13=_number.format('b13'), b14=_number.format('b14'), b15=_number.format('b15'), b16=_phone.format('b16'),
            c1=_binary.format('c1'), c2=_binary.format('c2'), c3=_number.format('c3'),
            d1=_gpos.format('d1'), d2=_number.format('d2'),
            e1=_gpos.format('e1'), e2=_number.format('e2'), e3=_number.format('e3'), e4=_number.format('e4'),
            e5=_number.format('e5'), e6=_number.format('e6'), e7=_number.format('e7'), e8=_number.format('e8'),
            f1=_gpos.format('f1'), f2=_number.format('f2'),
            g1=_number.format('g1'), g2=_number.format('g2'),
            h1=_number.format('h1'), h2=_number.format('h2'), h3=_number.format('h3'), h4=_number.format('h4'),
            h5=_regex.format('h5', pattern='.{0-6}'),
            i1=_number.format('i1'), i2=_number.format('i2'),
            j1=_number.format('j1'), j2=_number.format('j2'), j3=_number.format('j3'),
            state=_number.format('state')
        ))


class Label(object):
    """Container for full-context labels, allows for binarising of labels."""
    def __init__(self, file_path, state_level):
        """...

        :param file_path:
        :param state_level:
        """
        self.file_path = file_path
        self.lines = load_txt(self.file_path)

    def __repr__(self):
        return '\n'.join(self.lines)

    def __str__(self):
        return ' '.join(self.phones)

    def count_frames(self):
        """Counts the number of frames for each phone, i.e. the duration of each phone.

        Returns:
            (np.ndarray): Integer durations."""
        pass

    def binarise(self, question_set):
        """Queries the labels using the question set and returns vectorised labels.

        Args:
            question_file (str): File containing the question set.

        Returns:
            (np.ndarray): Binarised labels."""
        self.vector = np.array([question_set.query(line) for line in self.lines])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to load label files.")
    parser.add_argument("--lab_file", action="store", dest="lab_file", type=str,
                        help="Input label file.")
    add_arguments(parser)
    args = parser.parse_args()

    label = Label(args.lab_file)
    print(label)

