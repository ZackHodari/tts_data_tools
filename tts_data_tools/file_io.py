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
import tensorflow as tf


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


def make_SequenceExample(data, context=None):
    """Creates a `tf.train.SequenceExample` proto popoulated with the information in data.

    Args:
        data (dict<str,list<vector>>): A map of feature names to a sequence of frame-level vectors/floats/ints/strings.
        context (dict<str,vector>): A map of feature names to a vector/float/int/string."""
    def vector_to_Feature(vector):
        """Creates a `tf.train.Feature` proto."""
        if isinstance(vector, np.ndarray) and np.issubdtype(vector.dtype, np.integer):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=vector))
        elif isinstance(vector, np.ndarray) and np.issubdtype(vector.dtype, np.floating):
            return tf.train.Feature(float_list=tf.train.FloatList(value=vector))
        else:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(vector, 'utf8')]))

    def vectors_to_FeatureList(vectors):
        """Creates a `tf.train.FeatureList` proto."""
        return tf.train.FeatureList(feature=[
            vector_to_Feature(vector) for vector in vectors
        ])

    def data_to_FeatureLists(dictionary):
        """Creates a `tf.train.FeatureLists` proto."""
        return tf.train.FeatureLists(
            feature_list={
                key: vectors_to_FeatureList(vectors) for key, vectors in dictionary.items()
            }
        )

    def context_to_Features(dictionary):
        """Creates a `tf.train.Features` proto."""
        return tf.train.Features(
            feature={
                key: vector_to_Feature(vector) for key, vector in dictionary.items()
            }
        )

    if context is None:
        context = {}
    return tf.train.SequenceExample(feature_lists=data_to_FeatureLists(data),
                                    context=context_to_Features(context))


def load_lines(file_path):
    """Loads text data from a text file.

    Args:
        file_path (str): File to load the text from.

    Returns:
        (list<str>) Sequence of strings."""
    with open(file_path, 'r') as f:
        lines = list(filter(bool, map(str.strip, f.readlines())))

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
    sample_rate, data = wavfile.read(file_path)
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
        (`tf.train.SequenceExample`) Proto containing data for each feature."""
    with open(file_path, 'rb') as f:
        message = f.read()

    return tf.train.SequenceExample.FromString(message)


def save_proto(proto, file_path):
    """Converts data to a `tf.train.SequenceExample` proto, and serialises to binary.

    Args:
        data (dict<str,list<vector>>): A map of feature names to a sequence of frame-level vectors/floats/ints/string.
        file_path (str): File to save the data to.
        context (dict<str,vector>): A map of feature names to a vector/float/int/string."""
    message = proto.SerializeToString()

    with open(file_path, 'wb') as f:
        f.write(message)


def load_TFRecord(file_path):
    """Loads a list of `tf.train.SequenceExample` protos, saved in a `TFRecord`

    Args:
        file_path (str): File to load the data from.

    Returns:
        (list<tf.train.SequenceExample>) List of protos containing data for each feature."""
    record_iterator = tf.python_io.tf_record_iterator(path=file_path)

    protos = []
    for message in record_iterator:
        proto = tf.train.SequenceExample.FromString(message)
        protos.append(proto)

    return protos


def save_TFRecord(protos, file_path):
    """Saves a list of protos to a TFRecord, which can be used with `tf.data.TFRecordDataset`.

    Args:
        protos (list<tf.train.SequenceExample>) List of SequenceExample protos.
        file_path (str): File to load the data from."""
    # TODO(zackhodari): Add sharding.
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for proto in protos:
            message = proto.SerializeToString()
            writer.write(message)


def load_dataset(file_path, context_features, sequence_features, shapes, input_keys, target_keys,
                 max_examples=4096, batch_size=32):
    """Loads a TFRecord and parses the protos into a Tensorflow dataset, also shuffles and batches the data.

    NOTE: This function automatically adds the features `seq_len` from the proto. `seq_len` has shape [batch_size, 1].

    Usage:
    ```
        context_features = {
            'name': tf.FixedLenFeature((), tf.string)
        }

        sequence_features = {
            'lab': tf.FixedLenSequenceFeature(shape=[425], dtype=tf.float32),
            'f0': tf.FixedLenSequenceFeature(shape=[1], dtype=tf.float32),
        }

        input_shapes = {
            'name': [],
            'lab': [None, 425],
            'f0': [None, 1],
        }

        input_keys = ['name', 'lab', 'f0']
        target_keys = ['f0']

        train_dataset = load_dataset(file_path, context_features, sequence_features, shapes, input_keys, target_keys)
    ```

    Args:
        file_path (str): The name of the TFRecord file to load protos from.
        context_features (dict<str,feature_lens>): A dict containing sentence-level feature length specifications.
        sequence_features (dict<str,feature_lens>): A dict containing sequential feature length specifications.
        shapes (dict<list<int>>): A dict containing shape specifications.
        input_keys (list<str>): A list of keys that identify the features to be used as inputs.
        target_keys (list<str>): A list of keys that identify the features to be used as targets.
        max_examples (int): If specified, the dataset will be shuffled using `max_examples` samples of the full dataset.
        batch_size (int): Number of items in a batch.

    Return:
        (tf.data.TFRecordDataset) The padded and batched dataset.
    """
    raw_dataset = tf.data.TFRecordDataset(file_path)

    # Add sequence length to inputs as this will always be required.
    context_features['seq_len'] = tf.FixedLenFeature((1,), tf.int64)
    shapes['seq_len'] = [1]
    input_keys = list(input_keys) + ['seq_len']

    def _parse_proto(proto):
        context_dict, features_dict = tf.parse_single_sequence_example(proto, context_features, sequence_features)
        features_dict.update(context_dict)
        inputs = {key: features_dict[key] for key in input_keys}
        # targets = {key: features_dict[key] for key in target_keys}
        targets = tf.concat([features_dict[key] for key in target_keys], axis=-1)

        # TODO(zackhodari): Add mean-variance normalisation to dataset.
        return inputs, targets

    input_shapes = {key: shapes[key] for key in input_keys}
    # target_shapes = {key: shapes[key] for key in target_keys}
    target_shapes = [None, sum(shapes[key][1] for key in target_keys)]

    dataset = raw_dataset.map(_parse_proto)
    dataset = dataset.shuffle(max_examples)
    dataset = dataset.padded_batch(batch_size, padded_shapes=(input_shapes, target_shapes))
    dataset = dataset.prefetch(batch_size * 8)
    dataset = dataset.repeat()

    return dataset


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
    lines = load_lines(file_path)

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

