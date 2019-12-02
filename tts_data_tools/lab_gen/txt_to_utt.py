"""Handles generation of Utterance structures using Festival.

Usage:
    python txt_to_utt.py \
        --festival_dir DIR
        [--txt_file FILE]
        [--txt_dir DIR
         --id_list FILE]
        --out_file FILE
"""

import argparse
import os
import re
import subprocess

from tts_data_tools import file_io
from tts_data_tools import utils


def add_arguments(parser):
    parser.add_argument("--festival_dir", action="store", dest="festival_dir", type=str, required=True,
                        help="Directory containing festival installation.")
    parser.add_argument("--txt_file", action="store", dest="txt_file", type=str, default=None,
                        help="File containing all transcriptions.")
    parser.add_argument("--txt_dir", action="store", dest="txt_dir", type=str, default=None,
                        help="Directory containing text transcriptions.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, required=True,
                        help="List of file basenames to process (must be provided if txt_dir is used).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--custom_voice", action="store", dest="custom_voice", type=str, default=None,
                        help="Name of Festival voice to use when generating Utterance structures.")


def create_utterances(festival_dir, file_ids, sentences, out_dir, custom_voice=None):
    scm_commands = [f'#!{festival_dir}/bin/festival']

    if custom_voice is not None:
        # Run Festival with a particular voice.
        scm_commands.append(f'(voice_{custom_voice})')

    scm_command_str = '(utt.save (utt.synth (Utterance Text "{sentence}" )) "{utt_file}")'

    for file_id, sentence in zip(file_ids, sentences):
        utt_file = os.path.join(out_dir, 'utts', f'{file_id}.utt')

        scm_commands.append(scm_command_str.format(sentence=sentence, utt_file=utt_file))

    # Save the commands.
    gen_utts_scm_file = os.path.join(out_dir, 'gen_utts.scm')
    file_io.save_lines(scm_commands, gen_utts_scm_file)

    # If the file_ids are paths (e.g. for multi-speaker data), make sure the directory structure is already in place.
    utils.make_dirs(os.path.join(out_dir, 'utts'), file_ids)

    # Run the commands.
    festival_exe = os.path.join(festival_dir, 'bin', 'festival')
    scm_file = os.path.join(out_dir, 'gen_utts.scm')
    # Argument `check=True` ensures that an exception is raised if the process' return code is non-zero.
    subprocess.run([festival_exe, '-b', scm_file], check=True)


def process_file(festival_dir, txt_file, out_dir, custom_voice=None):
    """Create Utterance structures for all sentences in `txt_file` and save them to `out_dir`.

    Args:
        festival_dir (str): Directory containing festival installation.
        txt_file (str): File containing all transcriptions, with the following schema,
            (file_id, "sentence transcription")*
        out_dir (str): Directory to save the output to.
    """
    line_regex = re.compile(
        r'\(\s*'
        r'(?P<file_id>.+)'
        r'\s+'
        r'"(?P<sentence>.+)"'
        r'\s*\)')

    file_ids = []
    sentences = []

    # For all lines in txt_file extract file_id + sentence and add a command to create and save the Utterance structure.
    for line in file_io.load_lines(txt_file):

        match = re.match(line_regex, line)
        if match is None:
            print(f'Match not found for the following line,\n{line}')
            continue

        file_id = match.group('file_id')
        file_ids.append(file_id)

        sentence = match.group('sentence')
        sentence = sentence.replace('"', '\\"')
        sentences.append(sentence)

    # Save the file_ids.
    file_io.save_lines(file_ids, os.path.join(out_dir, 'file_id_list.scp'))

    # Create and save the Utterance structures.
    create_utterances(festival_dir, file_ids, sentences, out_dir, custom_voice=custom_voice)


def process_dir(festival_dir, txt_dir, id_list, out_dir, custom_voice=None):
    """Create Utterance structures for all sentences in `id_list` and save them to `out_dir`.

    Args:
        festival_dir (str): Directory containing festival installation.
        txt_dir (str): Directory containing text transcriptions.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
    """
    file_ids = utils.get_file_ids(id_list=id_list)

    sentences = []

    # For all file_ids load the sentence and add a command to create and save the Utterance structure.
    for file_id in sorted(file_ids):
        sentence = file_io.load_lines(os.path.join(txt_dir, f'{file_id}.txt'))[0]
        sentence = sentence.replace('"', '\\"')
        sentences.append(sentence)

    # If the file_ids are paths (e.g. for multi-speaker data), make sure the directory structure is already in place.
    utils.make_dirs(os.path.join(out_dir, 'utts'), file_ids)

    # Create and save the Utterance structures.
    create_utterances(festival_dir, file_ids, sentences, out_dir, custom_voice=custom_voice)


def process(festival_dir, txt_file=None, txt_dir=None, id_list=None, out_dir=None, custom_voice=None):
    if (txt_file is None) == (txt_dir is None):
        raise ValueError('Exactly one of txt_file or txt_dir must me specified.')

    if txt_dir is not None and id_list is None:
        raise ValueError('If txt_dir is used, id_list must be specified.')

    if txt_file is not None:
        process_file(festival_dir, txt_file, out_dir, custom_voice=custom_voice)

    if txt_dir is not None:
        process_dir(festival_dir, txt_dir, id_list, out_dir, custom_voice=custom_voice)


def main():
    parser = argparse.ArgumentParser(
        description="Generates Utterance structures from text files using Festival.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.festival_dir, args.txt_file, args.txt_dir, args.id_list, args.out_dir, args.custom_voice)


if __name__ == "__main__":
    main()

