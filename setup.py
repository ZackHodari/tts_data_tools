from setuptools import setup


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
        'pyreaper',
        'pysptk',
        'pyworld',
        'scipy',
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
        'tdt_lab_features = tts_data_tools.lab_features:main',
        'tdt_world = tts_data_tools.wav_gen.world:main',
        'tdt_reaper_f0 = tts_data_tools.wav_gen.reaper_f0:main',
        'tdt_world_with_reaper_f0 = tts_data_tools.wav_gen.world_with_reaper_f0:main',
        'tdt_mean_variance_normalisation = tts_data_tools.scripts.mean_variance_normalisation:main',
        'tdt_min_max_normalisation = tts_data_tools.scripts.min_max_normalisation:main',
        'tdt_process_dataset = tts_data_tools.scripts.process_dataset:main',
        'tdt_process_phones = tts_data_tools.scripts.process_phones:main',
    ]}
)


