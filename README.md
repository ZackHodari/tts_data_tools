# tts_data_tools
Data processing tools for preparing speech and labels for training TTS voices

## Installation
`pip install git+https://github.com/zackhodari/tts_data_tools`

## Usage

### Batch processing of datasets
```bash
tdt_process_dataset \
    --lab_dir DIR \
    --wav_dir DIR \
    --id_list FILE \
    --out_dir DIR \
    --question_file FILE \
    [--state_level | --no-state_level] \
    [--upsample_to_frame_level] \
    [--trim_silences] \
    [--subphone_feat_type STR] \
    [--calculate_normalisation] \
    [--normalisation_of_deltas]
```

Other batch processing scripts are defined in [setup.py](setup.py#L32-L45). Additional example scripts are given in [scripts/](tts_data_tools/scripts), these can be used from the command line or as templates for your own pre-processing.

### Label creation
[lab_gen](tts_data_tools/lab_gen) provides four tools to take text and convert them to numerical label features ready for training.

[txt_to_utt.py](tts_data_tools/lab_gen/txt_to_utt.py) and [utt_to_lab.py](tts_data_tools/lab_gen/utt_to_lab.py) wrap Festival (to be installed separately), and can extract full-context HTS-style labels.

[align_lab.py](tts_data_tools/lab_gen/align_lab.py) wraps HTK (to be installed separately) and performs forced alignment using the wavform.

If you already have forced-aligned labels you can use [lab_to_feat.py](tts_data_tools/lab_gen/lab_to_feat.py) to convert the text-formatted features to numpy arrays (saved as `.npy`):

```bash
tdt_lab_to_feat \
    --lab_dir DIR \
    --id_list FILE \
    --out_dir DIR \
    --question_file FILE \
    [--state_level | --no-state_level] \
    [--upsample_to_frame_level] \
    [--subphone_feat_type STR] \
    [--calculate_normalisation]
```

### Acoustic feature extraction
[wav_gen](tts_data_tools/wav_gen/) provides a few common vocoders that can be used to extract typical TTS acoustic features. This includes [F0 extraction with REAPER](tts_data_tools/wav_gen/reaper_f0.py) and full [vocoder feature extraction with WORLD](tts_data_tools/wav_gen/world.py), which can be used as follows.

```bash
tdt_world \
    --wav_dir DIR \
    --id_list FILE \
    --out_dir DIR \
    [--calculate_normalisation] \
    [--normalisation_of_deltas]
```
