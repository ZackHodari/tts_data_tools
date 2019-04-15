"""Runs the feature extraction on the waveforms and binarises the label files.

Usage:
    python process.py \
        [--lab_dir DIR] [--state_level] \
        [--wav_dir DIR] \
        [--id_list FILE] \
        --out_dir DIR
"""

import argparse
import os

from tts_data_tools import lab_features
from tts_data_tools import utils

from tts_data_tools.scripts.save_features import save_phones


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    lab_features.add_arguments(parser)


def extract_phones(file_ids, lab_dir, state_level):

    @utils.multithread(_lab_dir=lab_dir, _state_level=state_level)
    def extract(file_id, _lab_dir, _state_level):
        lab_path = os.path.join(_lab_dir, '{}.lab'.format(file_id))
        label = lab_features.Label(lab_path, _state_level)

        return label.phones

    return extract(file_ids)


def process(lab_dir, id_list, out_dir, state_level):
    """Processes label files in id_list, saves the phone identities (as a string) to text files.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
        """
    file_ids = utils.get_file_ids(lab_dir, id_list)

    phone_lists = extract_phones(file_ids, lab_dir, state_level)

    save_phones(file_ids, phone_lists, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract duration information from forced alignment label files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.id_list, args.out_dir, args.state_level)


if __name__ == "__main__":
    main()

