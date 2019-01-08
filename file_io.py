"""Handles loading and saving to files, using protobuffers, binary files, or text files.

Usage:
    python file_io.py --in_file FILE --out_file FILE
    python file_io.py --in_file FILE --out_file FILE --file_encoding ENUM
    python file_io.py --in_file FILE --out_file FILE --in_file_encoding ENUM --out_file_encoding ENUM"""

import argparse
from collections import Iterable
import enum
import re

import numpy as np
from tensorflow.train import Int64List, FloatList, Feature, FeatureList, FeatureLists, SequenceExample
from tensorflow.python_io import TFRecordWriter


FileEncodingEnum = enum.Enum("FileEncodingEnum", ("PROTO", "BIN", "TXT"))


def infer_file_encoding(file_ext):
    if file_ext in ['npy', 'lab', 'f0', 'lf0', 'sp', 'mgc', 'mfb', 'ap', 'bap']:
        file_ext = 'bin'
    if file_ext in ['dur']:
        file_ext = 'txt'
    return FileEncodingEnum[file_ext.upper()]


def add_arguments(parser):
    def enumtype(astring):
        try:
            return infer_file_encoding(astring)
        except KeyError:
            msg = ', '.join([t.name.lower() for t in FileEncodingEnum])
            msg = 'CustomEnumType: use one of {%s}'%msg
            raise argparse.ArgumentTypeError(msg)

    class OverrideAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.__setattr__(self.dest, values)

            # If file_encoding has already been written , then always use it to overwrite the other options.
            if namespace.file_encoding is not None:
                namespace.__setattr__('in_file_encoding', namespace.file_encoding)
                namespace.__setattr__('out_file_encoding', namespace.file_encoding)

    parser.register('action', 'override', OverrideAction)

    parser.add_argument("--in_file_encoding", action="override", dest="in_file_encoding", type=enumtype,
                        help="The encoding to load the file with.")
    parser.add_argument("--out_file_encoding", action="override", dest="out_file_encoding", type=enumtype,
                        help="The encoding to save the file with.")
    parser.add_argument("--file_encoding", action="override", dest="file_encoding", type=enumtype,
                        help="The encoding to load/save the file with. Overrides {in|out}_file_encoding arguments.")


def listify(values):
    if isinstance(values, Iterable):
        return list(values)
    else:
        return [values]


def load_proto(file_path):
    pass


def print_proto(file_path):
    pass


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
    """Converts data to a `tf.train.SequenceExample`, and saves it as a `TFRecord` file.

    Args:
        data (dict<str,list>): A map of feature names to a sequence of frame-level vectors/floats/ints.
        file_path (str): File to save the data to."""
    proto = make_SequenceExample(data)
    with TFRecordWriter(file_path) as writer:
        # write the example to a TFRecord file
        writer.write(proto.SerializeToString())


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
    array = np.array(data)

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

    return np.array(list(map(str.split, lines)), dtype=dtype)


def save_txt(data, file_path):
    """Saves data as a text file.

    If the data is floating point it is encoded into a string using scientific notation and 12 decimal places

    Args:
        data (np.ndarray): Sequence of frame-level vectors/floats/ints.
        file_path (str): File to save the data to."""
    array = np.array(data)

    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
        formatter = lambda x: '{:.12E}'.format(x)
    elif np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int32)
        formatter = lambda x: str(x)
    else:
        raise TypeError("Type of the data could not be serialised - {}".format(array.dtype))

    if array.ndim == 0:
        array = array[np.newaxis, np.newaxis]
    elif array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim != 2:
        raise ValueError("Only 1/2 dimensional data can be saved to text files, data.shape = {}".format(array.shape))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to load/save files in different encodings.")
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

