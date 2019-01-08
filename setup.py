from setuptools import setup

setup(
    name='tts_data_tools',
    version='0.1',
    description='Data processing tools for preparing speech and labels for training TTS voices.',
    url='https://github.com/ZackHodari/tts_data_tools',
    author='Zack Hodari',
    author_email='zack.hodari@ed.ac.uk',
    # license='MIT',
    packages=['tts_data_tools'],
    package_data={'tts_data_tools': ['question_sets/questions-unilex_dnn_600.hed',
                                     'question_sets/questions-radio_dnn_416.hed',
                                     'question_sets/questions-mandarin.hed',
                                     'question_sets/questions-japanese.hed']})
