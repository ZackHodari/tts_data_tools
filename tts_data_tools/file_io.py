"""Handles loading and saving to files, using protobuffers, binary files, or text files.

Usage:
    python file_io.py \
        --in_file FILE [--in_file_encoding ENUM] \
        --out_file FILE [--out_file_encoding ENUM]
"""

import argparse
from collections import Iterable
from enum import Enum
from pprint import pprint
from scipy.io import wavfile

import numpy as np
import pysptk
from tensorflow.python_io import TFRecordWriter
from tensorflow.train import Int64List, FloatList, Feature, FeatureList, FeatureLists, SequenceExample


FileEncodingEnum = Enum("FileEncodingEnum", ("PROTO", "BIN", "TXT"))


def infer_file_encoding(file_ext):
    """Converts file_ext to a FileEncodingEnum."""
    if file_ext in ['npy', 'lab', 'f0', 'lf0', 'sp', 'mgc', 'mfb', 'ap', 'bap']:
        file_ext = 'bin'
    if file_ext in ['dur']:
        file_ext = 'txt'
    return FileEncodingEnum[file_ext.upper()]


def add_arguments(parser):
    pass


def listify(values):
    if isinstance(values, Iterable):
        return list(values)
    else:
        return [values]


def sanitise_array(data):
    """Sanitises data to a numpy matrix of fixed shape, and ensures it has at most 2 axes.

    Args:
        data (list<_> or np.ndarray): Matrix/Vector/Scalar data in python lists (or as a numpy array).

    Returns:
        (np.ndarray): Sanitised numpy array with 2 axes."""
    array = np.array(data)

    if array.ndim == 0:
        array = array[np.newaxis, np.newaxis]
    elif array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim != 2:
        raise ValueError("Only 1/2 dimensional data can be saved to text files, data.shape = {}".format(array.shape))

    return array


def load_lines(file_path):
    """Loads text data from a text file.

    Args:
        file_path (str): File to load the text from.

    Returns:
        (list<str>) Sequence of strings."""
    with open(file_path, 'r') as f:
        lines = list(map(str.strip, f.readlines()))

    return lines


def save_lines(lines, file_path):
    """Saves text in a text file.

    Args:
        lines (list<str>): Sequence of strings.
        file_path (str): File to save the text to."""
    lines = list(map(lambda x: '{}\n'.format(line) for line in lines))

    with open(file_path, 'w') as f:
        f.writelines(lines)


def load_wav(file_path):
    """Loads wave data from wavfile.

    Args:
        file_path (str): File to load from.

    Returns:
        (np.ndarray) Waveform samples,
        (int) Sample rate of waveform."""
    sample_rate, data = wavfile.read(pysptk.util.example_audio_file())
    return data, sample_rate


def save_wav(file_path, data, sample_rate):
    """Saves wave data to wavfile.

    Args:
        data (np.ndarray): Waveform samples.
        file_path (str): File to save to."""
    wavfile.write(file_path, sample_rate, data)


def load_proto(file_path):
    """Loads data from a `tf.train.SequenceExample` proto.

    Args:
        file_path (str): File to load the data from.

    Returns:
        (dict<str, np.ndarray>) Dictionary containing data for each feature."""
    with open(file_path, 'rb') as f:
        message = f.read()

    return SequenceExample.FromString(message)


def print_proto(file_path):
    """Prints the data from a `tf.train.SequenceExample` proto.

    Args:
        file_path (str): File to load the data from."""
    proto = load_proto(file_path)
    pprint(proto)


def make_SequenceExample(data):
    """Creates a `tf.train.SequenceExample` proto popoulated with the information in data.

    Args:
        data (dict<str,list>): A map of feature names to a sequence of frame-level vectors/floats/ints."""
    def vector_to_Feature(vector):
        """Creates a `tf.train.Feature` proto."""
        vector = listify(vector)
        if isinstance(vector[0], int):
            return Feature(int64_list=Int64List(value=vector))
        else:
            return Feature(float_list=FloatList(value=vector))

    def vectors_to_FeatureList(vectors):
        """Creates a `tf.train.FeatureList` proto."""
        return FeatureList(feature=[
            vector_to_Feature(vector) for vector in vectors
        ])

    def dictionary_to_FeatureLists(dictionary):
        """Creates a `tf.train.FeatureLists` proto."""
        return FeatureLists(
            feature_list={
                key: vectors_to_FeatureList(vectors) for key, vectors in dictionary.items()
            }
        )

    return SequenceExample(feature_lists=dictionary_to_FeatureLists(data))


def save_proto(data, file_path):
    """Converts data to a `tf.train.SequenceExample` proto, and serialises to binary.

    Args:
        data (dict<str,list>): A map of feature names to a sequence of frame-level vectors/floats/ints.
        file_path (str): File to save the data to."""
    proto = make_SequenceExample(data)
    message = proto.SerializeToString()

    with open(file_path, 'wb') as f:
        f.write(message)


def load_bin(file_path, feat_dim, dtype=np.float32):
    """Loads data from a binary file using numpy.

    Args:
        file_path (str): File to load the data from.
        feat_dim (int): Dimensionality of the frame-level feature vectors.

    Returns:
        (np.ndarray) Sequence of frame-level vectors/floats/ints."""
    flat_data = np.fromfile(file_path, dtype=dtype)
    return flat_data.reshape((-1, feat_dim))


def save_bin(data, file_path):
    """Saves data as a binary file using numpy.

    Args:
        data (np.ndarray): Sequence of frame-level vectors/floats/ints.
        file_path (str): File to save the data to."""
    array = sanitise_array(data)

    if isinstance(array.dtype, np.floating):
        array = array.astype(np.float32)
    elif isinstance(array.dtype, np.integer):
        array = array.astype(np.int32)

    array.tofile(file_path)


def load_txt(file_path):
    """Loads data from a text file into a numpy array.

    Args:
        file_path (str): File to load the data from.

    Returns:
        (np.ndarray) Sequence of frame-level vectors/floats/ints."""
    with open(file_path, 'r') as f:
        lines = list(map(str.strip, f.readlines()))

    if 'E' in lines[0]:
        dtype = np.float32
    else:
        dtype = np.int32

    data = list(map(str.split, lines))
    array = np.array(data, dtype=dtype)
    return array


def save_txt(data, file_path):
    """Saves data as a text file.

    If the data is floating point it is encoded into a string using scientific notation and 12 decimal places.

    Args:
        data (np.ndarray): Sequence of frame-level vectors/floats/ints.
        file_path (str): File to save the data to."""
    array = sanitise_array(data)

    # If the data is floating then format the values in scientific notation.
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
        formatter = lambda x: '{:.12E}'.format(x)
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int32)
        formatter = lambda x: str(x)
    else:
        raise TypeError("Type of the data could not be serialised - {}".format(array.dtype))

    lines = [' '.join(formatter(val) for val in row) + '\n' for row in array]
    with open(file_path, 'w') as f:
        f.writelines(lines)


def load(file_path, file_encoding=None, feat_dim=None):
    if file_encoding is None:
        file_ext = file_path.split('.')[-1]
        file_encoding = infer_file_encoding(file_ext)

    if file_encoding == FileEncodingEnum.PROTO:
        return load_proto(file_path)

    elif file_encoding == FileEncodingEnum.BIN:
        return load_bin(file_path, feat_dim)

    elif file_encoding == FileEncodingEnum.TXT:
        return load_txt(file_path)

    else:
        raise NotImplementedError("Loading for file encoding '{}' not implemented".format(file_encoding))


def save(data, file_path, file_encoding=None):
    if file_encoding is None:
        file_ext = file_path.split('.')[-1]
        file_encoding = infer_file_encoding(file_ext)

    if file_encoding == FileEncodingEnum.PROTO:
        save_proto(data, file_path)

    elif file_encoding == FileEncodingEnum.BIN:
        save_bin(data, file_path)

    elif file_encoding == FileEncodingEnum.TXT:
        save_txt(data, file_path)

    else:
        raise NotImplementedError("Saving for file encoding '{}' not implemented".format(file_encoding))


def main():
    parser = argparse.ArgumentParser(description="Script to load/save files in different encodings.")

    def enumtype(astring):
        try:
            return infer_file_encoding(astring)
        except KeyError:
            msg = ', '.join([t.name.lower() for t in FileEncodingEnum])
            msg = 'CustomEnumType: use one of {%s}'%msg
            raise argparse.ArgumentTypeError(msg)

    parser.add_argument("--in_file_encoding", action="store", dest="in_file_encoding", type=enumtype,
                        help="The encoding to load the file with.")
    parser.add_argument("--out_file_encoding", action="store", dest="out_file_encoding", type=enumtype,
                        help="The encoding to save the file with.")

    parser.add_argument(
        "--in_file", action="store", dest="in_file", type=str, required=True, help="Input file.")
    parser.add_argument(
        "--out_file", action="store", dest="out_file", type=str, required=True, help="Output file.")
    parser.add_argument(
        "--feat_dim", action="store", dest="feat_dim", type=int, default=None, help="Dimensionality of binary feature.")
    add_arguments(parser)
    args = parser.parse_args()

    data = load(args.in_file, args.in_file_encoding, args.feat_dim)
    save(data, args.out_file, args.out_file_encoding)


if __name__ == "__main__":
    main()

