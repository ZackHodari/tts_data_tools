"""Handles loading and modifying label files.

Usage:
    python label_io.py --lab_file FILE"""

import argparse

import numpy as np

LABEL_DESCRIPTION = {
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

"""
1100000
1200000

xx~#-b+ei=s:1_4
p1ˆp2-p3+p4=p5 @p6 p7

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

class Label(object):
    """Container for full-context labels, allows for binarising of labels."""
    def __init__(self, file_path, state_level):
        """...

        :param file_path:
        :param state_level:
        """
        self.lines = load_txt(file_path)

    def __repr__(self):
        return '\n'.join(self.lines)

    def __str__(self):
        return ' '.join(self.phones)

    def count_frames(self):
        """Counts the number of frames for each phone, i.e. the duration of each phone.

        Returns:
            (np.ndarray): Integer durations."""
        pass

    def binarise(self, question_file):
        """Queries the labels using the question set and returns vectorised labels.

        Args:
            question_file (str): File containing the question set.

        Returns:
            (np.ndarray): Binarised labels."""
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to load label files.")
    parser.add_argument("--lab_file", action="store", dest="lab_file", type=str,
                        help="Input label file.")
    add_arguments(parser)
    args = parser.parse_args()

    label = Label(args.lab_file)
    print(label)

