from setuptools import setup

setup(
    name='tts_data_tools',
    version='0.1',
    description='Modular TensorFlow wrapper for training models on temporal data.',
    url='https://github.com/ZackHodari/tts_data_tools',
    author='Zack Hodari',
    author_email='zack.hodari@ed.ac.uk',
    # license='MIT',
    packages=['tts_data_tools'],
    data_files=[('tts_data_tools', ['tts_data_tools/question_sets/*'])])
