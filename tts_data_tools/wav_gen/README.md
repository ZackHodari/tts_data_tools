# Acoustic feature extraction
[wav_gen](wav_gen/) provides a few common vocoders that can be used to extract typical TTS acoustic features. This includes [F0 extraction with REAPER](wav_gen/reaper_f0.py) and full [vocoder feature extraction with WORLD](wav_gen/world.py), which can be used as follows.

```bash
tdt_world \
    --wav_dir DIR \
    --id_list FILE \
    --out_dir DIR \
    [--calculate_normalisation] \
    [--normalisation_of_deltas]
```
