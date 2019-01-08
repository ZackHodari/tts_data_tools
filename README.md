# tts_data_tools
Data processing tools for preparing speech and labels for training TTS voices

## Installation
`pip install git+https://github.com/zackhodari/tts_data_tools`

## Usage

### Batch processing of files
```bash
python process.py \
    [--lab_dir DIR] [--state_level] \
    [--wav_dir DIR] \
    [--id_list FILE] \
    --out_dir DIR
```


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
