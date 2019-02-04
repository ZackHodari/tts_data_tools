
import numpy as np
import tensorflow as tf


def arrays_to_SequenceExample(data, context=None):
    """Creates a `tf.train.SequenceExample` proto popoulated with the information in data.

    Args:
        data (dict<str,list<vector>>): A map of feature names to a sequence of frame-level vectors/floats/ints/strings.
        context (dict<str,vector>): A map of feature names to a vector/float/int/string."""
    if context is None:
        context = {}

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

    return tf.train.SequenceExample(feature_lists=data_to_FeatureLists(data),
                                    context=context_to_Features(context))


def SequenceExample_to_arrays(proto):
    """Converts a protobuffer into a dictionary of numpy arrays.

    Args:
        proto (tf.train.SequenceExample): The proto to convert.

    Returns:
         (dict<str,np.ndarray): A dictionary containing the data in numpy arrays.
    """
    def get_feature_value(feature):
        if feature.float_list.value:
            return feature.float_list.value
        if feature.int64_list.value:
            return feature.int64_list.value
        if feature.bytes_list.value:
            return [string.decode('utf-8') for string in feature.bytes_list.value]

    data = {}
    for key in proto.feature_lists.feature_list.keys():
        feature_list = proto.feature_lists.feature_list[key].feature
        data[key] = np.array([get_feature_value(feature) for feature in feature_list])

    context = {}
    for key in proto.context.feature.keys():
        feature = proto.context.feature[key]
        context[key] = np.array(get_feature_value(feature))

    return data, context


def load_dataset(file_path, context_features, sequence_features, shapes, input_keys, target_keys,
                 max_examples=4096, batch_size=32):
    """Loads a TFRecord and parses the protos into a Tensorflow dataset, also shuffles and batches the data.

    NOTE: This function automatically adds the `n_frames` & `n_phones` from the proto, both have shape [batch_size, 1].

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

    # Add sequence lengths to inputs as these will always be required.
    context_features['n_frames'] = tf.FixedLenFeature((1,), tf.int64)
    context_features['n_phones'] = tf.FixedLenFeature((1,), tf.int64)
    shapes['n_frames'] = [1]
    shapes['n_phones'] = [1]
    input_keys = list(input_keys) + ['n_frames', 'n_phones']

    def _parse_proto(proto):
        context_dict, features_dict = tf.parse_single_sequence_example(proto, context_features, sequence_features)
        features_dict.update(context_dict)
        inputs = {key: features_dict[key] for key in input_keys}
        # targets = {key: features_dict[key] for key in target_keys}
        targets = tf.concat([features_dict[key] for key in target_keys], axis=-1)

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

