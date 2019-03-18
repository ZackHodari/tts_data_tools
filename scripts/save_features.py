import os

from tts_data_tools.file_io import save_dir, save_lines, save_txt, save_bin


__all__ = ['save_phones', 'save_numerical_labels', 'save_counter_features',
           'save_durations', 'save_n_frames', 'save_n_phones',
           'save_lf0', 'save_vuv', 'save_sp', 'save_ap']


# Saved as a text files of strings.
def save_phones(file_ids, phone_lists, out_dir):
    path = os.path.join(out_dir, 'phones')

    save_dir(save_lines, path,
             phone_lists, file_ids, feat_ext='txt')


# Saved as a numpy binary file.
def save_numerical_labels(file_ids, numerical_labels, out_dir):
    path = os.path.join(out_dir, 'lab')
    save_dir(save_bin, path,
             numerical_labels, file_ids, feat_ext='lab')

    dim_path = os.path.join(out_dir, 'lab.dim')
    save_txt(numerical_labels[0].shape[1], dim_path)


# Saved as a numpy binary file.
def save_counter_features(file_ids, counter_features, out_dir):
    path = os.path.join(out_dir, 'counters')
    save_dir(save_bin, path,
             counter_features, file_ids, feat_ext='counters')

    dim_path = os.path.join(out_dir, 'counters.dim')
    save_txt(counter_features[0].shape[1], dim_path)


# Saved as a text file of integers.
def save_durations(file_ids, durations, out_dir):
    path = os.path.join(out_dir, 'dur')
    save_dir(save_txt, path,
             durations, file_ids, feat_ext='dur')

    dim_path = os.path.join(out_dir, 'dur.dim')
    save_txt(durations[0].shape[1], dim_path)


# Saved as a text file of integers.
def save_n_frames(file_ids, n_frames, out_dir):
    path = os.path.join(out_dir, 'n_frames')

    save_dir(save_txt, path,
             n_frames, file_ids, feat_ext='n_frames')


# Saved as a text file of integers.
def save_n_phones(file_ids, n_phones, out_dir):
    path = os.path.join(out_dir, 'n_phones')

    save_dir(save_txt, path,
             n_phones, file_ids, feat_ext='n_phones')


# Saved as a numpy binary file.
def save_lf0(file_ids, lf0_list, out_dir):
    path = os.path.join(out_dir, 'lf0')
    save_dir(save_bin, path,
             lf0_list, file_ids, feat_ext='lf0')

    dim_path = os.path.join(out_dir, 'lf0.dim')
    save_txt(lf0_list[0].shape[1], dim_path)


# Saved as a numpy binary file.
def save_vuv(file_ids, vuv_list, out_dir):
    path = os.path.join(out_dir, 'vuv')
    save_dir(save_bin, path,
             vuv_list, file_ids, feat_ext='vuv')

    dim_path = os.path.join(out_dir, 'vuv.dim')
    save_txt(vuv_list[0].shape[1], dim_path)


# Saved as a numpy binary file.
def save_sp(file_ids, sp_list, out_dir):
    path = os.path.join(out_dir, 'sp')
    save_dir(save_bin, path,
             sp_list, file_ids, feat_ext='sp')

    dim_path = os.path.join(out_dir, 'sp.dim')
    save_txt(sp_list[0].shape[1], dim_path)


# Saved as a numpy binary file.
def save_ap(file_ids, ap_list, out_dir):
    path = os.path.join(out_dir, 'ap')
    save_dir(save_bin, path,
             ap_list, file_ids, feat_ext='ap')

    dim_path = os.path.join(out_dir, 'ap.dim')
    save_txt(ap_list[0].shape[1], dim_path)

