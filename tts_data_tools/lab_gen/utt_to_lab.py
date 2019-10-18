"""Handles conversion from Festival Utterance structures to flat HTS-style full-context labels.

Usage:
    python utt_to_lab.py \
        --festival_dir DIR
        [--txt_file FILE]
        [--txt_dir DIR
         --id_list FILE]
        --out_file FILE
"""

import argparse
import os
import pkg_resources
import re
import subprocess

import numpy as np

from tts_data_tools import file_io
from tts_data_tools import utils

STATES_PER_PHONE = 5


def add_arguments(parser):
    parser.add_argument("--festival_dir", action="store", dest="festival_dir", type=str, required=True,
                        help="Directory containing festival installation.")
    parser.add_argument("--utt_dir", action="store", dest="utt_dir", type=str, required=True,
                        help="Directory containing Utterance structures.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, required=True,
                        help="List of file basenames to process (must be provided if txt_dir is used).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--extra_feats_scm", action="store", dest="extra_feats_scm", type=str, default='extra_feats.scm',
                        help="Directory to save the output to.")
    parser.add_argument("--label_feats", action="store", dest="label_feats", type=str, default='label.feats',
                        help="Directory to save the output to.")
    parser.add_argument("--label_full_awk", action="store", dest="label_full_awk", type=str, default='label-full.awk',
                        help="Directory to save the output to.")
    parser.add_argument("--label_mono_awk", action="store", dest="label_mono_awk", type=str, default='label-mono.awk',
                        help="Directory to save the output to.")


def utts_to_dumps(dumpfeats_exe, utt_dir, file_ids, dump_dir,
                  extra_feats_scm='extra_feats.scm', label_feats='label.feats'):

    if extra_feats_scm in pkg_resources.resource_listdir('tts_data_tools', os.path.join('resources', 'festival')):
        print(f'Using tts_data_tools resource from resources/festival for {extra_feats_scm}')
        extra_feats_scm = pkg_resources.resource_filename('tts_data_tools',
                                                          os.path.join('resources', 'festival', extra_feats_scm))

    if label_feats in pkg_resources.resource_listdir('tts_data_tools', os.path.join('resources', 'festival')):
        print(f'Using tts_data_tools resource from resources/festival for {label_feats}')
        label_feats = pkg_resources.resource_filename('tts_data_tools',
                                                      os.path.join('resources', 'festival', label_feats))

    utils.make_dirs(dump_dir, file_ids)

    for file_id in file_ids:
        # Argument `check=True` ensures that an exception is raised if the process' return code is non-zero.
        subprocess.run([dumpfeats_exe,
                        '-eval', extra_feats_scm,
                        '-relation', 'Segment',
                        '-feats', label_feats,
                        '-output', os.path.join(dump_dir, f'{file_id}.txt'),
                        os.path.join(utt_dir, f'{file_id}.utt')], check=True)


def dumps_to_labs(dump_dir, file_ids, label_out_dir, awk='label-full.awk'):

    if awk in pkg_resources.resource_listdir('tts_data_tools', os.path.join('resources', 'festival')):
        print(f'Using tts_data_tools resource from resources/festival for {awk}')
        awk = pkg_resources.resource_filename('tts_data_tools', os.path.join('resources', 'festival', awk))

    utils.make_dirs(label_out_dir, file_ids)

    for file_id in file_ids:
        # Argument `check=True` ensures that an exception is raised if the process' return code is non-zero.
        rtn = subprocess.run(['gawk', '-f', awk, os.path.join(dump_dir, f'{file_id}.txt')],
                             check=True, stdout=subprocess.PIPE)

        # `stdout` was redirected with a pipe and stored in the return object `rtn` as a binary string.
        with open(os.path.join(label_out_dir, f'{file_id}.lab'), 'wb') as f:
            f.write(rtn.stdout)


def mark_silences(label, current_phone_regex=re.compile('-(.+?)\+'), n_gram_size=2, is_mono=False):
    phones = []
    is_silence = []

    for line in label:
        if is_mono:
            phone = line

        else:
            match = current_phone_regex.search(line)

            if match is None:
                raise ValueError(f'Regex failed for line,\n{line}')

            phone = match.group(1)

        phones.append(phone)
        is_silence.append(phone in ['pau', 'sil'])

    first_phone_idx = is_silence.index(False)
    last_phone_idx = len(is_silence) - 1 - is_silence[::-1].index(False)

    # Replace the pauses before and after the phone sequence with silences.
    for i, line in enumerate(label):
        if is_mono:
            if i < first_phone_idx + n_gram_size or i > last_phone_idx - n_gram_size:
                label[i] = line.replace('pau', 'sil')

        else:
            n_gram = line[:line.index('/A')]
            remaining_label = line[line.index('/A'):]

            if i < first_phone_idx + n_gram_size or i > last_phone_idx - n_gram_size:
                label[i] = n_gram.replace('pau', 'sil') + remaining_label

    return label


def round_dur(dur):
    return int(round(dur / 50000, 0) * 50000)


def sanitise_labs(lab_dir, file_ids, label_out_dir, include_times=False, state_level=False, is_mono=False):

    utils.make_dirs(label_out_dir, file_ids)

    for file_id in file_ids:
        label = file_io.load_lines(os.path.join(lab_dir, f'{file_id}.lab'))
        n_phones = len(label)

        start_times, end_times, label = map(list, zip(*map(str.split, label)))
        label = mark_silences(label, is_mono=is_mono)

        if state_level:
            if include_times:
                n_states = n_phones * STATES_PER_PHONE

                times = np.interp(range(0, n_states + 1, 1),
                                  range(0, n_states + 1, STATES_PER_PHONE),
                                  start_times + end_times[-1:])

                start_times = times[:-1]
                end_times = times[1:]

            label = np.repeat(label, STATES_PER_PHONE).tolist()
            for i in range(len(label)):
                state_idx = i % STATES_PER_PHONE
                label[i] += f'[{state_idx+2}]'

        if include_times:
            start_times = list(map(round_dur, start_times))
            end_times = list(map(round_dur, end_times))

            label = list(map(' '.join, zip(*[start_times, end_times, label])))

        file_io.save_lines(label, os.path.join(label_out_dir, f'{file_id}.lab'))


def process(festival_dir, utt_dir, id_list, out_dir,
            extra_feats_scm='extra_feats.scm', label_feats='label.feats',
            label_full_awk='label-full.awk', label_mono_awk='label-mono.awk'):
    """Create flat HTS-style full-context labels.

    Args:
        festival_dir (str): Directory containing festival installation.
        utt_dir (str): Directory containing Utterance structures.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        extra_feats_scm (str): .
        label_feats (str): .
        label_full_awk (str): .
        label_mono_awk (str): .
    """
    file_ids = utils.get_file_ids(id_list=id_list)

    dumpfeats_exe = os.path.join(festival_dir, 'examples', 'dumpfeats')

    label_dump_dir = os.path.join(out_dir, 'label_phone_align', 'dump')
    label_full_dir = os.path.join(out_dir, 'label_phone_align', 'full')
    label_mono_dir = os.path.join(out_dir, 'label_phone_align', 'mono')
    label_no_align_dir = os.path.join(out_dir, 'label_no_align')
    mono_no_align_dir = os.path.join(out_dir, 'mono_no_align')

    # Create the flattened features and format them according to `label_full_awk` and `label_mono_awk`.
    utts_to_dumps(dumpfeats_exe, utt_dir, file_ids, label_dump_dir, extra_feats_scm, label_feats)
    dumps_to_labs(label_dump_dir, file_ids, label_full_dir, label_full_awk)
    dumps_to_labs(label_dump_dir, file_ids, label_mono_dir, label_mono_awk)

    # Clean up the full-context label features: replace initial pauses with 'sil' and remove timestamps.
    sanitise_labs(label_full_dir, file_ids, label_no_align_dir, include_times=False, state_level=False)
    sanitise_labs(label_mono_dir, file_ids, mono_no_align_dir, include_times=False, state_level=False, is_mono=True)


def main():
    parser = argparse.ArgumentParser(
        description="Flattens Festival Utterance structures into HTS full-context labels.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.festival_dir, args.utt_dir, args.id_list, args.out_dir,
            args.extra_feats_scm, args.label_feats, args.label_full_awk, args.label_mono_awk)


if __name__ == "__main__":
    main()

