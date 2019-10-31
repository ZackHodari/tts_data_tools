from setuptools import setup


setup(
    name='tts_data_tools',
    version='0.5',
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
        'resources/question_sets/questions-unilex_dnn_600.hed',
        'resources/question_sets/questions-unilex_phones_69.hed',
        'resources/question_sets/questions-radio_dnn_416.hed',
        'resources/question_sets/questions-radio_phones_48.hed',
        'resources/question_sets/questions-mandarin.hed',
        'resources/festival/extra_feats.scm',
        'resources/festival/label.feats',
        'resources/festival/label-full.awk',
        'resources/festival/label-mono.awk',
    ]},
    entry_points={'console_scripts': [
        'tdt_lab_gen = tts_data_tools.lab_gen:main',
        'tdt_txt_to_utt = tts_data_tools.lab_gen.txt_to_utt:main',
        'tdt_utt_to_lab = tts_data_tools.lab_gen.utt_to_lab:main',
        'tdt_align_lab = tts_data_tools.lab_gen.align_lab:main',
        'tdt_lab_to_feat = tts_data_tools.lab_gen.lab_to_feat:main',
        'tdt_world = tts_data_tools.wav_gen.world:main',
        'tdt_reaper_f0 = tts_data_tools.wav_gen.reaper_f0:main',
        'tdt_world_with_reaper_f0 = tts_data_tools.wav_gen.world_with_reaper_f0:main',
        'tdt_mean_variance_normalisation = tts_data_tools.scripts.mean_variance_normalisation:main',
        'tdt_min_max_normalisation = tts_data_tools.scripts.min_max_normalisation:main',
        'tdt_process_dataset = tts_data_tools.scripts.process_dataset:main',
        'tdt_process_phones = tts_data_tools.scripts.process_phones:main',
    ]}
)

