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
import glob
import os
import random
import re
import subprocess

from tts_data_tools import file_io
from tts_data_tools.lab_gen import utils
from tts_data_tools.utils import get_file_ids, make_dirs

from tts_data_tools.scripts.mean_variance_normalisation import calculate_mvn_parameters

# String constants for various shell calls.
STATES_PER_PHONE = 5
F = str(0.01)
SFAC = str(5.0)
PRUNING = [str(i) for i in (250., 150., 2000.)]

MACROS = 'macros'
HMMDEFS = 'hmmdefs'
VFLOORS = 'vFloors'


def add_arguments(parser):
    parser.add_argument("--htk_dir", action="store", dest="htk_dir", type=str, required=True,
                        help="Directory containing HTK installation.")
    parser.add_argument("--lab_dir", action="store", dest="lab_dir", type=str, required=True,
                        help="Directory containing HTS-style state-level labels without alignments.")
    parser.add_argument("--wav_dir", action="store", dest="wav_dir", type=str, required=True,
                        help="Directory containing the wavfiles.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, required=True,
                        help="List of file basenames to process.")
    parser.add_argument("--out_dir", action="store", dest="out_dir", type=str, required=True,
                        help="Directory to save the output to.")
    parser.add_argument("--multiple_speaker", action="store_true", dest="multiple_speaker", default=False,
                        help="Whether the data contains multiple speakers.")
    parser.add_argument("--num_train_proccesses", action="store", dest="num_train_proccesses", type=str, default=4,
                        help="Number of parallel processes to use for HMM training.")


class ForcedAlignment(object):

    def __init__(self, htk_dir, lab_dir, wav_dir, id_list, out_dir):
        self.HCompV = os.path.join(htk_dir, 'HCompV')
        self.HCopy = os.path.join(htk_dir, 'HCopy')
        self.HERest = os.path.join(htk_dir, 'HERest')
        self.HHEd = os.path.join(htk_dir, 'HHEd')
        self.HVite = os.path.join(htk_dir, 'HVite')

        self.wav_dir = wav_dir
        self.lab_dir = lab_dir

        self.file_ids = get_file_ids(id_list=id_list)
        self.file_ids = self.check_file_ids(id_list)

        print('---preparing environment')

        # Directories
        # -----------

        self.cfg_dir = os.path.join(out_dir, 'config')
        self.model_dir = os.path.join(out_dir, 'model')
        self.cur_dir = os.path.join(self.model_dir, 'hmm0')
        self.mfc_dir = os.path.join(out_dir, 'mfc')
        self.mono_lab_dir = os.path.join(out_dir, 'mono_no_align')

        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.cur_dir, exist_ok=True)
        os.makedirs(self.mfc_dir, exist_ok=True)
        os.makedirs(self.mono_lab_dir, exist_ok=True)

        # Paths
        # -----

        self.phonemes = os.path.join(out_dir, 'mono_phone.list')
        self.phoneme_map = os.path.join(out_dir, 'phoneme_map.dict')
        self.align_mlf = os.path.join(out_dir, 'mono_align.mlf')

        # HMMs
        self.proto = os.path.join(self.cfg_dir, 'proto')

        # SCP files
        self.copy_scp = os.path.join(self.cfg_dir, 'copy.scp')
        self.train_scp = os.path.join(self.cfg_dir, 'train.scp')
        self.phoneme_mlf = os.path.join(self.cfg_dir, 'mono_phone.mlf')

        # CFG
        self.cfg = os.path.join(self.cfg_dir, 'cfg')

    def check_file_ids(self, file_ids):
        validated_file_ids = []

        for file_id in file_ids:
            wav_file = os.path.join(self.wav_dir, f'{file_id}.wav')
            lab_file = os.path.join(self.lab_dir, f'{file_id}.lab')

            if os.path.exists(wav_file) and os.path.exists(lab_file):
                validated_file_ids.append(file_id)

        return validated_file_ids

    def prepare_data(self, multiple_speaker=False):
        print('---preparing data')
        _, _, mfc_paths = self.make_scp(self.file_ids)
        if multiple_speaker:
            speaker_names = self.get_speaker_names(self.file_ids)
            mfc_paths_by_speaker = self.split_by_speaker(speaker_names, mfc_paths)

        else:
            mfc_paths_by_speaker = {'all': mfc_paths}

        print('---extracting features')
        self.full_to_mono(self.file_ids)
        self.compute_mfccs()

        print('---feature_normalisation')
        for speaker_mfc_paths in mfc_paths_by_speaker.items():
            self.normalise_inplace(speaker_mfc_paths)

        print('---making proto')
        self.make_proto()

    def make_scp(self, file_ids):
        wav_paths = []
        lab_paths = []
        mfc_paths = []

        for file_id in file_ids:
            wav_paths.append(os.path.join(self.wav_dir, f'{file_id}.wav'))
            lab_paths.append(os.path.join(self.lab_dir, f'{file_id}.lab'))

            # HVite requires a flat directory structure, so mfc files use the base_name of file_id.
            base_name = os.path.basename(file_id)
            mfc_paths.append(os.path.join(self.mfc_dir, f'{base_name}.mfc'))

        file_io.save_lines(map(' '.join, zip(wav_paths, mfc_paths)), self.copy_scp)
        file_io.save_lines(mfc_paths, self.train_scp)

        return wav_paths, lab_paths, mfc_paths

    def get_speaker_names(self, file_ids):
        speaker_names = []

        for file_id in file_ids:
            tmp_list = file_id.split('/')

            speaker_name = tmp_list[0]
            speaker_names.append(speaker_name)

        return speaker_names

    def split_by_speaker(self, speaker_names, paths):
        paths_by_speaker = {speaker_name: [] for speaker_name in set(speaker_names)}

        for speaker_name, file_path in zip(speaker_names, paths):
            paths_by_speaker[speaker_name].append(file_path)

        return paths_by_speaker

    def full_to_mono(self, file_ids):
        phone_set = set()

        for file_id in file_ids:
            base_name = os.path.basename(file_id)
            lab_file = os.path.join(self.lab_dir, f'{file_id}.lab')
            # HVite requires a flat directory structure, so mono-lab files use the base_name of file_id.
            mono_lab_file = os.path.join(self.mono_lab_dir, f'{base_name}.lab')

            phones = self._full_to_mono(lab_file, mono_lab_file)
            phone_set.update(phones)

        file_io.save_lines(phone_set, self.phonemes)
        file_io.save_lines(map(' '.join, zip(phone_set, phone_set)), self.phoneme_map)

        with open(self.phoneme_mlf, 'w') as f:
            f.write('#!MLF!#\n')
            f.write(f'"*/*.lab" => "{self.mono_lab_dir}"\n')

    def _full_to_mono(self, full_file_name, mono_file_name, current_phone_regex=re.compile('-(.+?)\+')):
        phones = []

        label = file_io.load_lines(full_file_name)
        for line in label:
            phone = current_phone_regex.search(line).group(1)
            phones.append(phone)

        file_io.save_lines(phones, mono_file_name)

        return phones

    def compute_mfccs(self):
        """
        Compute MFCCs
        """
        # Write a CFG for extracting MFCCs.
        with open(self.cfg, 'w') as f:
            f.write('SOURCEKIND = WAVEFORM\n'
                    'SOURCEFORMAT = WAVE\n'
                    'TARGETRATE = 50000.0\n'
                    'TARGETKIND = MFCC_D_A_0\n'
                    'WINDOWSIZE = 250000.0\n'
                    'PREEMCOEF = 0.97\n'
                    'USEHAMMING = T\n'
                    'ENORMALIZE = T\n'
                    'CEPLIFTER = 22\n'
                    'NUMCHANS = 20\n'
                    'NUMCEPS = 12')

        subprocess.run([self.HCopy, '-C', self.cfg, '-S', self.copy_scp], check=True)

        # Write a CFG for what we just built.
        with open(self.cfg, 'w') as f:
            f.write('TARGETRATE = 50000.0\n'
                     'TARGETKIND = USER\n'
                     'WINDOWSIZE = 250000.0\n'
                     'PREEMCOEF = 0.97\n'
                     'USEHAMMING = T\n'
                     'ENORMALIZE = T\n'
                     'CEPLIFTER = 22\n'
                     'NUMCHANS = 20\n'
                     'NUMCEPS = 12')

    def normalise_inplace(self, file_paths):
        data_list = []

        for file_path in file_paths:

            with utils.open_htk(file_path, 'rb') as f:
                data, n_samples = f.read_all()
                data_list.append(data)

        # Compute mean and variance.
        mvn_params, _ = calculate_mvn_parameters(data_list)

        # Normalise the data and save in HTK format.
        for file_path, data in zip(file_paths, data_list):
            norm_data = (data - mvn_params['mean']) / mvn_params['std_dev']

            with utils.open_htk(file_path, 'wb') as f:
                f.write_all(norm_data)

    def make_proto(self):
        # make proto
        means = ' '.join(['0.0' for _ in range(39)])
        vars = ' '.join(['1.0' for _ in range(39)])

        with open(self.proto, 'w') as f:
            f.write('~o <VECSIZE> 39 <USER>\n'
                    '~h "proto"\n'
                    '<BEGINHMM>\n'
                    '<NUMSTATES> 7')

            for i in range(2, STATES_PER_PHONE + 2):
                f.write(f'<STATE> {i}\n<MEAN> 39\n{means}\n')
                f.write(f'<VARIANCE> 39\n{vars}\n')

            f.write('<TRANSP> 7\n'
                    ' 0.0 1.0 0.0 0.0 0.0 0.0 0.0\n'
                    ' 0.0 0.6 0.4 0.0 0.0 0.0 0.0\n'
                    ' 0.0 0.0 0.6 0.4 0.0 0.0 0.0\n'
                    ' 0.0 0.0 0.0 0.6 0.4 0.0 0.0\n'
                    ' 0.0 0.0 0.0 0.0 0.6 0.4 0.0\n'
                    ' 0.0 0.0 0.0 0.0 0.0 0.7 0.3\n'
                    ' 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n'
                    '<ENDHMM>')

        # Make vFloors
        subprocess.run([self.HCompV,
                        '-f', F,
                        '-C', self.cfg,
                        '-S', self.train_scp,
                        '-M', self.cur_dir,
                        self.proto], check=True)

        # Make local macro.
        with open(os.path.join(self.cur_dir, MACROS), 'w') as f:

            # Get first three lines from local proto.
            with open(os.path.join(self.cur_dir, 'proto'), 'r') as source:
                for _ in range(3):
                    f.write(source.readline())

            # Get remaining lines from vFloors.
            with open(os.path.join(self.cur_dir, VFLOORS), 'r') as source:
                f.writelines(source.readlines())

        # Make hmmdefs.
        with open(os.path.join(self.cur_dir, HMMDEFS), 'w') as f:

            with open(self.proto, 'r') as source:
                # Ignore first two lines.
                source.readline()
                source.readline()

                source_lines = source.readlines()

            phone_set = file_io.load_lines(self.phonemes)
            for phone in phone_set:
                # The header.
                f.write(f'~h "{phone}"\n')

                # The rest.
                f.writelines(source_lines)

    def train_hmm(self, niter, num_mix, num_splits=1):
        """
        Perform one or more rounds of estimation
        """
        print('---training HMM models')

        if num_splits != 1:
            # Call HERest in multiple chunks, split scp in num_splits chunks and save them.
            print(f'----num_splits set to {num_splits}')

            train_scp_chunks = []

            with open(self.train_scp, "rt") as fp:
                mfc_files = fp.readlines()
            random.shuffle(mfc_files)

            n = (len(mfc_files) + 1) // num_splits
            mfc_chunks = [mfc_files[j:j + n] for j in range(0, len(mfc_files), n)]

            for i, mfc_chunk in enumerate(mfc_chunks):
                train_scp_chunk = os.path.join(self.cfg_dir, f'train_{i}.scp')
                train_scp_chunks.append(train_scp_chunk)

                file_io.save_lines(mfc_chunk, train_scp_chunk)

        done = 0
        mix = 1
        while mix <= num_mix and done == 0:
            for i in range(niter):
                next_dir = os.path.join(self.model_dir, f'hmm_mix_{mix}_iter_{i+1}')
                if not os.path.exists(next_dir):
                    os.makedirs(next_dir)

                if num_splits == 1:
                    subprocess.run(
                        [self.HERest,
                         '-C', self.cfg,
                         '-S', self.train_scp,
                         '-I', self.phoneme_mlf,
                         '-M', next_dir,
                         '-H', os.path.join(self.cur_dir, MACROS),
                         '-H', os.path.join(self.cur_dir, HMMDEFS),
                         '-t', *PRUNING,
                         self.phonemes],
                        stdout=subprocess.PIPE,
                        check=True)
                else:
                    procs = []
                    # Estimate per chunk.
                    for chunk_num in range(len(train_scp_chunks)):
                        procs.append(subprocess.Popen(
                            [self.HERest,
                             '-C', self.cfg,
                             '-S', train_scp_chunks[chunk_num],
                             '-I', self.phoneme_mlf,
                             '-M', next_dir,
                             '-H', os.path.join(self.cur_dir, MACROS),
                             '-H', os.path.join(self.cur_dir, HMMDEFS),
                             '-t', *PRUNING,
                             '-p', str(chunk_num + 1),
                             self.phonemes],
                            stdout=subprocess.PIPE))

                    # Wait until all HERest calls are finished.
                    for p in procs:
                        p.wait()

                    # Now accumulate.
                    subprocess.run(
                        [self.HERest,
                         '-C', self.cfg,
                         '-M', next_dir,
                         '-H', os.path.join(self.cur_dir, MACROS),
                         '-H', os.path.join(self.cur_dir, HMMDEFS),
                         '-t', *PRUNING,
                         '-p', '0',
                         self.phonemes,
                         *glob.glob(next_dir + os.sep + "*.acc")],
                        stdout=subprocess.PIPE,
                        check=True)

                self.cur_dir = next_dir

            if mix * 2 <= num_mix:
                # Increase mixture number.
                hed_file = os.path.join(self.cfg_dir, f'mix_{mix * 2}.hed')
                with open(hed_file, 'w') as f:
                    f.write(f'MU {mix * 2} {{*.state[2-{STATES_PER_PHONE + 2}].mix}}\n')

                next_dir = os.path.join(self.model_dir, f'hmm_mix_{mix * 2}_iter_0')
                os.makedirs(next_dir, exist_ok=True)

                subprocess.run(
                    [self.HHEd, '-A',
                     '-H', os.path.join(self.cur_dir, MACROS),
                     '-H', os.path.join(self.cur_dir, HMMDEFS),
                     '-M', next_dir,
                     hed_file,
                     self.phonemes],
                    check=True)

                self.cur_dir = next_dir
                mix *= 2

            else:
                done = 1

    def align(self, lab_align_dir, lab_dir, file_ids):
        """
        Align using the models in self.cur_dir and MLF to palab_align_dirth
        """
        print('---aligning data')

        subprocess.run(
            [self.HVite, '-a', '-f', '-m',
             '-y', 'lab',
             '-o', 'SM',
             '-i', self.align_mlf,
             '-L', self.mono_lab_dir,
             '-C', self.cfg,
             '-S', self.train_scp,
             '-H', os.path.join(self.cur_dir, MACROS),
             '-H', os.path.join(self.cur_dir, HMMDEFS),
             '-I', self.phoneme_mlf,
             '-t', *PRUNING,
             '-s', SFAC,
             self.phoneme_map,
             self.phonemes],
            check=True)

        print('Checking alignment MLF before running\n'
              f'\tForcedAlignment()._postprocess({self.align_mlf}, {lab_align_dir}, {lab_dir}, "<file_ids>")')

        self._check_alignments_present(self.align_mlf, file_ids)
        self._add_alignments_to_lab(self.align_mlf, lab_align_dir, lab_dir, file_ids)

    def _check_alignments_present(self, mlf, file_ids, base_name_regex=re.compile(r'"([\w/]+(\.[\w/]+)*)"')):
        base_names = list(map(os.path.basename, file_ids))

        with open(mlf, 'r') as f:
            # Consume the MLF #!header!# line.
            _ = f.readline()

            mlf_ids = []
            while True:
                line = f.readline().strip()
                match = re.match(base_name_regex, line)

                # Read lines until we reach one containing the name of a file.
                if match is not None:
                    label_path = os.path.basename(match.group(1))
                    label_base_name = os.path.splitext(label_path)[0]
                    mlf_ids.append(label_base_name)

                # Reached the end of the file.
                if len(line) < 1:
                    break

        base_names = set(base_names)
        mlf_ids = set(mlf_ids)

        err_str = 'Alignment output error'

        missing_from_mlf = base_names.difference(mlf_ids)
        if len(missing_from_mlf) > 0:
            err_str += '\nFollowing files are missing from alignment MLF, it is likely that alignment failed for them\n'
            err_str += '\n'.join(missing_from_mlf)

        missing_from_id_list = mlf_ids.difference(base_names)
        if len(missing_from_id_list) > 0:
            err_str += '\nFollowing files are missing from id_list, but alignments were generated, use a full id_list\n'
            err_str += '\n'.join(missing_from_id_list)

        if len(missing_from_mlf) > 0 or len(missing_from_id_list) > 0:
            raise ValueError(err_str)

    def _add_alignments_to_lab(self, mlf, lab_align_dir, lab_dir, file_ids):
        make_dirs(lab_align_dir, file_ids)

        with open(mlf, 'r') as f:
            # Consume the MLF #!header!# line.
            _ = f.readline()

            for file_id in file_ids:
                # Consume the file name line.
                line = f.readline()

                mlf_base_name = os.path.splitext(os.path.basename(line))[0]
                id_base_name = os.path.basename(file_id)

                if mlf_base_name != id_base_name:
                    raise ValueError(f'The file order in the mlf ({mlf}) does not match file_ids)\n'
                                     f'{mlf_base_name} {id_base_name}')

                label_no_align = file_io.load_lines(os.path.join(lab_dir, f'{file_id}.lab'))

                label_state_align = []
                for label_tag in label_no_align:
                    label_tag = label_tag.strip()

                    for i in range(STATES_PER_PHONE):
                        # Consume a state alignment line.
                        line = f.readline().strip()

                        # Get the alignments for this state.
                        start_time, end_time, *_ = line.split()
                        label_state_align.append(f'{start_time} {end_time} {label_tag}[{i + 2}]')

                # label_state_align
                file_io.save_lines(label_state_align, os.path.join(lab_align_dir, f'{file_id}.lab'))

                # Consume the end of file line marker ('.' character).
                line = f.readline().strip()

                if line != '.':
                    raise ValueError('The two files are not matched!')


def process(htk_dir, lab_dir, wav_dir, id_list, out_dir, multiple_speaker=False, num_train_proccesses=1):
    """Create flat HTS-style full-context labels.

    Args:
        htk_dir (str): Directory containing HTK installation.
        lab_dir (str): Directory containing HTS-style state-level labels without alignments.
        wav_dir (str): Directory containing the wavfiles.
        id_list (str): List of file basenames to process.
        out_dir (str): Directory to save the output to.
        multiple_speaker (bool): Whether the data contains multiple speakers.
        num_train_proccesses (int): Number of parallel processes to use for HMM training.
    """
    aligner = ForcedAlignment(htk_dir, lab_dir, wav_dir, id_list, out_dir)

    # After `ForcedAlignment.check_file_ids` some files may be excluded.
    file_ids = aligner.file_ids

    aligner.prepare_data(multiple_speaker)
    aligner.train_hmm(7, 32, num_splits=num_train_proccesses)
    aligner.align(os.path.join(out_dir, 'label_state_align'), lab_dir, file_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Flattens Festival Utterance structures into HTS full-context labels.")
    add_arguments(parser)
    args = parser.parse_args()

    process(args.htk_dir, args.lab_dir, args.wav_dir, args.id_list, args.out_dir,
            args.multiple_speaker, args.num_train_proccesses)


if __name__ == "__main__":
    main()

