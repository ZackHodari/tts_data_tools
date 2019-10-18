import argparse


def add_arguments(parser):
    parser.add_argument("--state_level", dest="state_level", action="store_true", default=True,
                        help="Is the label file state level (or frame level).")
    parser.add_argument("--no-state_level", dest="state_level", action="store_false", help=argparse.SUPPRESS)


def main():
    import sys

    from tts_data_tools.lab_gen import (
        txt_to_utt,
        utt_to_lab,
        align_lab,
        lab_to_feat)

    all_args = list(sys.argv[1:])

    # Text to Utterance structures.
    parser = argparse.ArgumentParser(description="Process text files.")
    txt_to_utt.add_arguments(parser)
    args, _ = parser.parse_known_args(all_args)

    txt_to_utt.process(
        args.festival_dir, args.txt_file, args.txt_dir, args.id_list, args.out_dir)

    # Utterance to lab.
    parser = argparse.ArgumentParser(description="Process Utterance files.")
    utt_to_lab.add_arguments(parser)
    args, _ = parser.parse_known_args(all_args)

    utt_to_lab.process(
        args.festival_dir, args.utt_dir, args.id_list, args.out_dir,
        args.extra_feats_scm, args.label_feats, args.label_full_awk, args.label_mono_awk)

    # Align labels.
    parser = argparse.ArgumentParser(description="Align label files.")
    align_lab.add_arguments(parser)
    args, _ = parser.parse_known_args(all_args)

    align_lab.process(
        args.htk_dir, args.lab_dir, args.wav_dir, args.id_list, args.out_dir,
        args.multiple_speaker, args.num_train_proccesses)

    # Label to numerical features.
    parser = argparse.ArgumentParser(description="Process label files.")
    lab_to_feat.add_arguments(parser)
    args, _ = parser.parse_known_args(all_args)

    lab_to_feat.process(
        args.lab_dir, args.id_list, args.out_dir, args.state_level,
        args.question_file, args.upsample_to_frame_level, args.subphone_feat_type, args.calculate_normalisation)


if __name__ == "__main__":
    main()

