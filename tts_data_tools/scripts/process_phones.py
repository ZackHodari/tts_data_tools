"""Extracts phonetic identities from label files.

Usage:
    python process.py \
        [--lab_dir DIR] [--state_level] \
        [--wav_dir DIR] \
        [--id_list FILE] \
        --out_dir DIR
"""

import argparse
import os

from tts_data_tools import file_io
from tts_data_tools import lab_gen
from tts_data_tools import utils
from tts_data_tools.lab_gen import lab_to_feat


def add_arguments(parser):
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to be converted.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    lab_gen.add_arguments(parser)


def process(lab_dir, id_list, out_dir, state_level):
    """Processes label files in id_list, saves the phone identities (as a string) to text files.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
    """
    file_ids = utils.get_file_ids(id_list=id_list)

    for file_id in file_ids:
        # Label processing.
        lab_path = os.path.join(lab_dir, '{}.lab'.format(file_id))
        label = lab_to_feat.Label(lab_path, state_level)

        phones = label.phones
        n_phones = len(label.phones)

        file_io.save_lines(phones, os.path.join(out_dir, 'phones', '{}.txt'.format(file_id)))
        file_io.save_txt(n_phones, os.path.join(out_dir, 'n_phones', '{}.txt'.format(file_id)))


def main():
    parser = argparse.ArgumentParser(
        description="Extracts phonetic identities from label files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.id_list, args.out_dir, args.state_level)


if __name__ == "__main__":
    main()

