from setuptools import setup

setup(
    name='tts_data_tools',
    version='0.2',
    description='Data processing tools for preparing speech and labels for training TTS voices.',
    url='https://github.com/ZackHodari/tts_data_tools',
    author='Zack Hodari',
    author_email='zack.hodari@ed.ac.uk',
    # license='MIT',
    packages=['tts_data_tools'],
    package_data={'tts_data_tools': ['question_sets/questions-unilex_dnn_600.hed',
                                     'question_sets/questions-radio_dnn_416.hed',
                                     'question_sets/questions-mandarin.hed',
                                     'question_sets/questions-japanese.hed']},
    entry_points={'console_scripts': ['tdt_process = tts_data_tools.process:main',
                                      'tdt_file_io = tts_data_tools.file_io:main',
                                      'tdt_lab_features = tts_data_tools.lab_features:main',
                                      'tdt_wav_features = tts_data_tools.wav_features:main']})
