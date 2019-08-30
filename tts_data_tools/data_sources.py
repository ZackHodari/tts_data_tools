"""Consistent interface for loading/saving files. All _DataSources produce data that can be used in ML libraries."""

import os

import numpy as np

from tts_data_tools import file_io
from tts_data_tools import utils
from tts_data_tools.wav_gen.utils import compute_deltas


class _DataSource(object):
    r"""Abstract data loading class.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : None or str
        Type of normalisation to perform. This allows the type of normalisation to be specified, but the normaliser
        itself will not be contained within the data source, that must be handled outside of the data source.
    use_deltas : bool
        Whether to compute delta features.
    ext : str, optional
        The file extension of the saved features, if not set `self.name` is used.

    Notes
    -----
    The data setup assumes a folder structure such as the following example,

    .. code-block::

        dataset_name (data_root)

            train (data_dir)

                lab (name)
                    *.lab
                lab_minmax.json

                lf0 (name)
                    *.npy
                lf0_mvn.json
                lf0_deltas_mvn.json
                ...

            valid (data_dir)
                ...

            ...

    All data is contained below `data_root`.

    There can be multiple `data_dir` directories, e.g. one for each data split (train, valid, test).

    Each feature should have a directory within `data_dir`, this will contain all files for this feature.

    While normalisation is not handled here, you should ensure there are files present containing the normalisation
    parameters, e.g. 'lf0_mvn.json'. Such files should exist for all data sources requiring normalisation, with an
    additional file for all data sources using delta features.
    """
    def __init__(self, name, normalisation=None, use_deltas=False, ext=None):
        self.name = name
        self.normalisation = normalisation
        self.use_deltas = use_deltas
        self.ext = ext if ext is not None else name

    def file_path(self, base_name, data_dir):
        r"""Creates file path for a given base name and data directory."""
        return os.path.join(data_dir, self.name, '{}.{}'.format(base_name, self.ext))

    def load_files(self, base_names, data_dir):
        for base_name in base_names:
            yield self.load_file(base_name, data_dir)

    def save_files(self, data, base_names, data_dir):
        utils.make_dirs(data_dir, base_names)

        for datum, base_name in zip(data, base_names):
            self.save_file(datum, base_name, data_dir)

    def load_file(self, base_name, data_dir):
        r"""Loads the contents of a given file. Must either be a sequence feature with 2 dimensions or a scalar value.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        int or float or bool or `np.ndarray`, shape (seq_len, feat_dim)
        """
        raise NotImplementedError

    def save_file(self, data, base_name, data_dir):
        r"""Saves data to a file using the format defined by the class.

        Parameters
        ----------
        data : int or float or bool or `np.ndarray`, shape (seq_len, feat_dim)
            Data loaded from the file specified.
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.
        """
        raise NotImplementedError

    def __call__(self, base_name, data_dir):
        r"""Loads the feature and creates deltas if specified by this data source.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        dict[str, (int or float or bool or np.ndarray)]
            Loaded feature, and deltas if specified.
        """
        feature = self.load_file(base_name, data_dir)
        features = {self.name: feature}

        if self.use_deltas:
            deltas = compute_deltas(feature)
            features['{}_deltas'.format(self.name)] = deltas.astype(np.float32)

        return features


class StringSource(_DataSource):
    r"""Loads data from a text file, this will be loaded as strings where each item should be on a new line.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    ext : str, optional
        The file extension of the saved features, if not set `name` is used.
    """
    def __init__(self, name, ext='txt'):
        super(StringSource, self).__init__(name, normalisation=None, use_deltas=False, ext=ext)

    def load_file(self, base_name, data_dir):
        r"""Loads lines of text.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        list<str>
        """
        file_path = self.file_path(base_name, data_dir)
        return file_io.load_lines(file_path)

    def save_file(self, data, base_name, data_dir):
        r"""Saves text as a text file.

        Parameters
        ----------
        data : list<str>
            Sequence of strings.
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.
        """
        file_path = self.file_path(base_name, data_dir)
        file_io.save_lines(data, file_path)


class ASCIISource(StringSource):
    r"""Loads data from a text file, this will be loaded as strings where each item should be on a new line.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    ext : str, optional
        The file extension of the saved features, if not set `name` is used.
    """
    def __init__(self, name, ext='txt'):
        super(ASCIISource, self).__init__(name, ext=ext)

    def load_file(self, base_name, data_dir):
        r"""Loads the lines and converts to ASCII codes (np.int8), each line is considered as a sequence item.

        Each line can have a different number of characters, the maximum number of characters will be used to determine
        the shape of the 2nd dimension of the array.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        np.ndarray, shape (seq_len, max_num_characters), dtype (np.int8)
        """
        lines = super(ASCIISource, self).load_file(base_name, data_dir)

        # Convert the strings into ASCII integers. Padding is also partially handled here.
        feature = utils.string_to_ascii(lines)
        return feature

    def save_file(self, data, base_name, data_dir):
        r"""Saves ASCII codes as a text file.

        Parameters
        ----------
        data : np.ndarray, shape (seq_len, max_num_characters), dtype (np.int8)
            Sequence of strings stored as ASCII codes (and padded with \x00).
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.
        """
        lines = utils.ascii_to_string(data)
        super(ASCIISource, self).save_file(lines, base_name, data_dir)


class WavSource(_DataSource):
    r"""Loads wavfiles using `scipy.io.wavfile`.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : str
        Type of normalisation to perform. This allows the type of normalisation to be specified, but the normaliser
        itself will not be contained within the data source, that must be handled outside of the data source.
    use_deltas : bool
        Whether to compute delta features. If normalisation is being used it will also perform normalisation of deltas.

    Attributes
    ----------
    sample_rate : int
        The sample rate of the wavfiles being loaded, if not given this will be set in `self.load_file`.
    """
    def __init__(self, name, normalisation=None, use_deltas=False, sample_rate=None):
        super(WavSource, self).__init__(name, normalisation, use_deltas, ext='wav')

        self.sample_rate = sample_rate

    def load_file(self, base_name, data_dir):
        r"""Loads a wavfile using `scipy.io.wavfile`.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        np.ndarray, shape (num_samples,), dtype (np.int16)
        """
        file_path = self.file_path(base_name, data_dir)
        sample_rate, feature = file_io.load_wav(file_path)

        if self.sample_rate is None:
            self.sample_rate = sample_rate

        return feature.reshape(-1, 1)

    def save_file(self, data, base_name, data_dir):
        r"""Saves the feature as a wavfile using scipy.io.wavfile.write

        Parameters
        ----------
        data : int or float or bool or `np.ndarray`, shape (seq_len, feat_dim)
            Data loaded from the file specified.
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.
        """
        file_path = self.file_path(base_name, data_dir)
        file_io.save_wav(data, file_path, self.sample_rate)


class NumpyBinarySource(_DataSource):
    r"""Data loading class for features saved with `np.ndarray.tofile`, loading is thus performed using `np.fromfile`.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : str
        Type of normalisation to perform. This allows the type of normalisation to be specified, but the normaliser
        itself will not be contained within the data source, that must be handled outside of the data source.
    use_deltas : bool
        Whether to compute delta features. If normalisation is being used it will also perform normalisation of deltas.
    ext : str, optional
        The file extension of the saved features, if not set `name` is used.
    """
    def __init__(self, name, normalisation=None, use_deltas=False, ext='npy'):
        super(NumpyBinarySource, self).__init__(name, normalisation, use_deltas, ext)

    def load_file(self, base_name, data_dir):
        r"""Loads the feature using `np.load`.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        int or float or bool or np.ndarray, shape (seq_len, feat_dim)
        """
        file_path = self.file_path(base_name, data_dir)
        return file_io.load_bin(file_path)

    def save_file(self, data, base_name, data_dir):
        r"""Saves the feature using `np.save`.

        Parameters
        ----------
        data : int or float or bool or `np.ndarray`, shape (seq_len, feat_dim)
            Data loaded from the file specified.
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.
        """
        file_path = self.file_path(base_name, data_dir)
        file_io.save_bin(data, file_path)


class TextSource(_DataSource):
    r"""Loads data from a text file, this can contain integers or floats and will have up to 2 dimensions.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : str
        Type of normalisation to perform. This allows the type of normalisation to be specified, but the normaliser
        itself will not be contained within the data source, that must be handled outside of the data source.
    use_deltas : bool
        Whether to compute delta features. If normalisation is being used it will also perform normalisation of deltas.
    ext : str, optional
        The file extension of the saved features, if not set `name` is used.
    """
    def __init__(self, name, normalisation=None, use_deltas=False, ext='txt'):
        super(TextSource, self).__init__(name, normalisation, use_deltas, ext)

    def load_file(self, base_name, data_dir):
        r"""Loads the feature from a text file into a numpy array.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        int or float or np.ndarray, shape (seq_len, feat_dim)
        """
        file_path = self.file_path(base_name, data_dir)
        feature = file_io.load_txt(file_path)

        # If the sequence length feature is describing a sentence level length, convert it to a scalar.
        if feature.shape[0] == 1:
            feature = feature.item()

        return feature

    def save_file(self, data, base_name, data_dir):
        r"""Saves data as a text file.

        Parameters
        ----------
        data : int or float or bool or `np.ndarray`, shape (seq_len, feat_dim)
            Data loaded from the file specified.
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.
        """
        file_path = self.file_path(base_name, data_dir)
        file_io.save_txt(data, file_path)

