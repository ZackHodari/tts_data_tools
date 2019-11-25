# Label creation
[lab_gen](lab_gen) provides four tools to take text and convert them to numerical label features ready for training.

[txt_to_utt.py](lab_gen/txt_to_utt.py) and [utt_to_lab.py](lab_gen/utt_to_lab.py) wrap Festival (to be installed separately), and can extract full-context HTS-style labels.

[align_lab.py](lab_gen/align_lab.py) wraps HTK (to be installed separately) and performs forced alignment using the wavform.

If you already have forced-aligned labels you can use [lab_to_feat.py](lab_gen/lab_to_feat.py) to convert the text-formatted features to numpy arrays (saved as `.npy`):

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
