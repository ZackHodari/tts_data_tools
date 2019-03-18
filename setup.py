from setuptools import setup

setup(
    name='tts_data_tools',
    version='0.3',
    description='Data processing tools for preparing speech and labels for training TTS voices.',
    url='https://github.com/ZackHodari/tts_data_tools',
    author='Zack Hodari',
    author_email='zack.hodari@ed.ac.uk',
    # license='MIT',
    packages=['tts_data_tools'],
    package_data={'tts_data_tools': [
        'question_sets/questions-unilex_dnn_600.hed',
        'question_sets/questions-unilex_phones_69.hed',
        'question_sets/questions-radio_dnn_416.hed',
        'question_sets/questions-radio_phones_48.hed',
        'question_sets/questions-mandarin.hed',
        'question_sets/questions-japanese.hed'
    ]},
    entry_points={'console_scripts': [
        'tdt_file_io = tts_data_tools.file_io:main',
        'tdt_lab_features = tts_data_tools.lab_features:main',
        'tdt_wav_features = tts_data_tools.wav_features:main',
        'tdt_extract_counters = scripts.extract_counters:main',
        'tdt_extract_durations = scripts.extract_durations:main',
        'tdt_extract_lf0_and_vuv_reaper = scripts.extract_lf0_and_vuv_reaper:main',
        'tdt_extract_numerical_labels = scripts.extract_numerical_labels:main',
        'tdt_extract_numerical_labels_and_durations = scripts.extract_numerical_labels_and_durations:main',
        'tdt_extract_phones = scripts.extract_phones:main',
        'tdt_extract_world = scripts.extract_world:main',
        'tdt_mean_variance_normalisation = scripts.mean_variance_normalisation:main',
        'tdt_min_max_normalisation = scripts.min_max_normalisation:main',
        'tdt_process_acoustics = scripts.process_acoustics:main',
        'tdt_process_dataset = scripts.process_dataset:main',
        'tdt_process_labels = scripts.process_labels:main'
    ]}
)

