import argparse

from tts_data_tools.wav_gen import utils


def add_arguments(parser):
    parser.add_argument("--type", action="store", dest="type", type=str, required=True,
                        help="Name of the waveform generation module to use.")


def main():
    parser = argparse.ArgumentParser(description="Process waveforms or features.")
    add_arguments(parser)
    args, remaining_args = parser.parse_known_args()

    # Import and run the chosen waveform generation module.
    eval('from tts_data_tools.wav_gen import {}'.format(args.type))
    module = eval(args.type)
    module.main()


if __name__ == "__main__":
    main()

