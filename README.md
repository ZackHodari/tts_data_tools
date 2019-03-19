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
    --out_dir DIR \
    --question_file FILE \
    [--id_list FILE] \
    [--state_level | --no-state_level] \
    [--upsample_to_frame_level] \
    [--subphone_feat_type STR] \
    [--calculate_normalisation] \
    [--normalisation_of_deltas]
```

Other batch processing scripts are given in [scripts](scripts). These can be used from the command line with the prefix `tdt_`, or as templates to create your own processing.


### Modifying file encodings
```bash
python file_io.py \
    --in_file FILE [--in_file_encoding ENUM] \
    --out_file FILE [--out_file_encoding ENUM]
```
```bash
python file_io.py \
    --in_file FILE \
    --out_file FILE \
    --file_encoding ENUM
```

### Label normalisation on individual files
```bash
python lab_features.py \
    --lab_file FILE [--state_level] \
    --question_file FILE [--subphone_feat_type STR] \
    --out_file FILE
```

### Feature extraction on individual files
```bash
python wav_features.py \
    --wav_file FILE \
    --out_file FILE
```
