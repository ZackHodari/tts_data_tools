"""Parses sentences into segments with the pattern '{chink*, chunk*}*'.

From Church (1992)
- https://www.phon.ucl.ac.uk/courses/plin/plin2108/docs/liberman-church-1992.pdf
- pp. 8 "Text analysis and word pronunciation in text-to-speech synthesis", Mark Liberman and Kenneth Church (1992)
- pp. 791-832 in Furui and Sondhi, Eds., Advances in Speech Technology, Marcel Dekker

chink
-----
- Function (closed) words and tensed verbs
    - IN (prepositions)
    - DT, PDT (determiners)
    - PRP$, WP$ (pronouns) (excluded PRP)
    - MD (auxiliary verbs)
    - CC (conjunctions)
    - RP (particles)
    - (numerals)
    - (qualifiers/intensifiers)
    - (interrogatives)
    - EX, TO, UH, OF (OF is added by Festival)

- VBD, VBP, VBZ (finite/tensed verb forms)

- PRP_sub not in [her him me them us] (subjective personal pronoun) (i.e. [it you])
    it you
    its your
    one

chunk
-----
- Content (open) and objective pronouns
    - NN, NNS, NNP, NNPS (nouns)
    - VB, VBG, VBN (verbs) (excluded VBD, VBP, VBZ)
    - JJ, JJR, JJS (adjectives)
    - RB, RBR, RBS, WRB (adverbs)
    - CD (numbers)
    - FW (foreign word)
    - WDT, WP (where) (this could go in either category)
    - POS (should be replaced with previous tag, most likely noun?)

- PRP_obj in [her him me them us] (objective personal pronoun)
    her him me them us
    hers his
    she he
    we they
    our ours
    my mine

LS	    List item marker (bullet points)
SYM	    Symbol (punctuation)

Usage:
    python chinks_and_chunks_splitter.py \
        --lab_dir DIR [--state_level] \
        --id_list FILE \
        --out_dir DIR
"""

import argparse
import numpy as np
import os
import re

from tts_data_tools import file_io
from tts_data_tools import lab_gen
from tts_data_tools import utils
from tts_data_tools.lab_gen import lab_to_feat
from tts_data_tools.wav_gen import world_with_reaper_f0

OBJECTIVE_PRONOUNS = ('her', 'him', 'me', 'them', 'us', 'hers', 'his', 'she', 'he', 'we',
                      'they', 'our', 'ours', 'my', 'mine')

CHINK_TAGS = ('in', 'dt', 'pdt', 'prp$', 'wp$', 'md', 'cc', 'rp', 'ex', 'to',
              'uh', 'of', 'vbd', 'vbp', 'vbz', 'prp_sub')

CHUNK_TAGS = ('nn', 'nns', 'nnp', 'nnps', 'vb', 'vbg', 'vbn', 'jj', 'jjr', 'jjs',
              'rb', 'rbr', 'rbs', 'wrb', 'cd', 'fw', 'wdt', 'wp', 'pos',
              'prp_obj')


def add_arguments(parser):
    parser.add_argument("--lab_dir_with_pos", action="store", dest="lab_dir_with_pos", type=str, required=True,
                        help="Directory of the label files containing POS tags.")
    parser.add_argument("--wav_dir", action="store", dest="wav_dir", type=str, required=True,
                        help="Directory of the wav files.")
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory of the label files to take alignments from.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to process (must be contained in lab_dir).")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    lab_gen.add_arguments(parser)


def get_word_idxs(label, word_idx_sep=(r'@', r'\+'), phrase_idx_sep=(r'@', r'=')):

    word_idx_regex = re.compile(fr'(?:{word_idx_sep[0]}(?P<word_idx>\d+?){word_idx_sep[1]})')
    phrase_idx_regex = re.compile(fr'(?:{phrase_idx_sep[0]}(?P<phrase_idx>\d+?){phrase_idx_sep[1]})')

    # Maintain a unique word_id that gets reset at silences, accumulate phone indices for each word.
    word_id = None
    phone_idxs = []
    for i, line in enumerate(label):
        word_idx_match = re.search(word_idx_regex, line)
        phrase_idx_match = re.search(phrase_idx_regex, line)

        # Silence.
        if word_idx_match is None or phrase_idx_match is None:
            # Ignore first silence.
            if word_id is None:
                continue

            # Setting this to None will ensure that a `New word` is started below.
            word_id = None
            phone_idxs[-1].append(i)

        # Phone.
        else:
            word_idx = int(word_idx_match.group('word_idx'))
            phrase_idx = int(phrase_idx_match.group('phrase_idx'))

            # New word.
            if word_id is None or word_id != (word_idx, phrase_idx):
                word_id = (word_idx, phrase_idx)
                phone_idxs.append([])

            phone_idxs[-1].append(i)

    # Ignore last silence.
    if word_id is None:
        phone_idxs[-1].pop()

    word_start_idxs = [phone_idx[0] for phone_idx in phone_idxs]
    word_end_idxs = [phone_idx[-1] + 1 for phone_idx in phone_idxs]

    return word_start_idxs, word_end_idxs


def get_pos_tags(label, word_start_idxs, pos_sep=(r'/E:', r'\+'), surface_form_sep=(r'\+', r'_'),
                 objective_pronouns=OBJECTIVE_PRONOUNS):

    pos_regex = re.compile(fr'{pos_sep[0]}(?P<pos_tag>.{{1,4}}?){pos_sep[1]}')
    surface_form_regex = re.compile(fr'{surface_form_sep[0]}(?P<surface_form>.*?){surface_form_sep[1]}')

    pos_tags = []
    for word_idx in word_start_idxs:
        pos_match = re.search(pos_regex, label[word_idx])
        surface_form_match = re.search(surface_form_regex, label[word_idx])
        if pos_match is None or surface_form_match is None:  # Silences.
            pos_tags.append('silence')
            continue

        pos_tag = pos_match.group('pos_tag')

        if pos_tag == 'prp':

            surface_form = surface_form_match.group('surface_form')

            if surface_form.lower() in objective_pronouns:
                pos_tag = 'prp_obj'
            else:
                pos_tag = 'prp_sub'

        pos_tags.append(pos_tag)

    return pos_tags


def segment_words(word_start_idxs, word_end_idxs, pos_tags,
                  chink_tags=CHINK_TAGS, chunk_tags=CHUNK_TAGS):

    chink_or_chunk = []
    for pos_tag in pos_tags:

        if pos_tag == 'ls':
            chink_or_chunk.append('bullet')

        elif pos_tag in chink_tags:
            chink_or_chunk.append('chink')

        elif pos_tag in chunk_tags:
            chink_or_chunk.append('chunk')

        elif pos_tag in ['sym', 'silence']:
            chink_or_chunk.append('punctuation')

        else:
            raise ValueError(f'POS tag ({pos_tag}) encountered that is not in chink/chunk lists.')

    segment_regex = re.compile(r'(?: ?bullet)?(?: ?chink)*(?: ?chunk)*(?: ?punctuation)?')
    segments = list(filter(bool, re.findall(segment_regex, ' '.join(chink_or_chunk))))
    segment_word_lens = list(map(len, map(str.split, segments)))  # Empty strings are excluded by str.split

    i = 0
    segment_start_idxs = []
    segment_end_idxs = []
    for segment_word_len in segment_word_lens:
        segment_phone_start_idx = word_start_idxs[i]
        i += segment_word_len
        segment_phone_end_idx = word_end_idxs[i - 1]

        segment_start_idxs.append(segment_phone_start_idx)
        segment_end_idxs.append(segment_phone_end_idx)

    return segment_start_idxs, segment_end_idxs


def process(lab_dir, id_list, out_dir, state_level, lab_dir_with_pos, wav_dir):
    """Processes label files in id_list, saves the phone identities (as a string) to text files.

    Args:
        lab_dir (str): Directory containing the label files.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        state_level (bool): Indicates that the label files are state level if True, otherwise they are frame level.
    """
    file_ids = utils.get_file_ids(id_list=id_list)

    utils.make_dirs(os.path.join(out_dir, 'segment_n_phones'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'segment_n_frames'), file_ids)
    utils.make_dirs(os.path.join(out_dir, 'n_segments'), file_ids)

    for file_id in file_ids:
        lab_path_with_pos = os.path.join(lab_dir_with_pos, f'{file_id}.lab')
        label_with_pos = file_io.load_lines(lab_path_with_pos)

        word_start_idxs, _ = get_word_idxs(
            label_with_pos, word_idx_sep=(r'@', r'\+'), phrase_idx_sep=(r'@', r'='))
        pos_tags = get_pos_tags(label_with_pos, word_start_idxs)

        lab_path = os.path.join(lab_dir, f'{file_id}.lab')
        label = lab_to_feat.Label(lab_path, state_level)

        durations = label.phone_durations
        n_frames = np.sum(durations).item()
        n_phones = len(label.phones)

        word_start_idxs, word_end_idxs = get_word_idxs(
            label.labels, word_idx_sep=(r':', r'\+'), phrase_idx_sep=(r':', r'='))
        try:
            segment_start_idxs, segment_end_idxs = segment_words(word_start_idxs, word_end_idxs, pos_tags)
        except (ValueError, IndexError) as e:
            print(f'{e}\n{file_id}')
        else:
            wav_path = os.path.join(wav_dir, f'{file_id}.wav')
            wav, sample_rate = file_io.load_wav(wav_path)
            f0, _, _, _ = world_with_reaper_f0.analysis(wav, sample_rate)

            # Match the number of frames between label forced-alignment and vocoder analysis.
            # Often the durations from forced alignment are a few frames longer than the vocoder features.
            diff = n_frames - f0.shape[0]
            if diff > n_phones:
                raise ValueError(f'Number of label frames and vocoder frames is too different for {file_id}\n'
                                 f'\tlabel frames {n_frames}\n'
                                 f'\tvocoder frames {f0.shape[0]}\n'
                                 f'\tnumber of phones {n_phones}')

            # Remove excess durations if there is a shape mismatch.
            if diff > 0:
                # Remove 1 frame from each phone's duration starting at the end of the sequence.
                durations[-diff:] -= 1
                n_frames = f0.shape[0]
                print(f'Cropped {diff} frames from durations for utterance {file_id}')

            assert n_frames == np.sum(durations).item()

            segment_phone_lens = []
            segment_frame_lens = []
            for segment_start_idx, segment_end_idx in zip(segment_start_idxs, segment_end_idxs):
                segment_phone_lens.append(segment_end_idx - segment_start_idx)
                segment_frame_lens.append(sum(durations[segment_start_idx:segment_end_idx]))

            file_io.save_txt(segment_phone_lens, os.path.join(out_dir, 'segment_n_phones', f'{file_id}.txt'))
            file_io.save_txt(segment_frame_lens, os.path.join(out_dir, 'segment_n_frames', f'{file_id}.txt'))
            file_io.save_txt(len(segment_phone_lens), os.path.join(out_dir, 'n_segments', f'{file_id}.txt'))


def main():
    parser = argparse.ArgumentParser(
        description="Extracts phonetic identities from label files.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.lab_dir, args.id_list, args.out_dir, args.state_level, args.lab_dir_with_pos, args.wav_dir)


if __name__ == "__main__":
    main()

