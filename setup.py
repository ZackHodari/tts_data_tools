from setuptools import setup

script_def = 'tdt_{name} = tts_data_tools.scripts.{name}:main'

setup(
    name='tts_data_tools',
    version='0.3',
    description='Data processing tools for preparing speech and labels for training TTS voices.',
    url='https://github.com/ZackHodari/tts_data_tools',
    author='Zack Hodari',
    author_email='zack.hodari@ed.ac.uk',
    # license='MIT',
    install_requires=[
        'numpy',
        'pyworld',
        'pyreaper',
        'tqdm'
    ],
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
        script_def.format(name='extract_counters'),
        script_def.format(name='extract_durations'),
        script_def.format(name='extract_lf0_and_vuv_reaper'),
        script_def.format(name='extract_numerical_labels'),
        script_def.format(name='extract_numerical_labels_and_durations'),
        script_def.format(name='extract_phones'),
        script_def.format(name='extract_world'),
        script_def.format(name='mean_variance_normalisation'),
        script_def.format(name='min_max_normalisation'),
        script_def.format(name='process_acoustics'),
        script_def.format(name='process_dataset'),
        script_def.format(name='process_labels')
    ]}
)


