# FOF dataset on DANDI

Scripts that pack the calcium imaging dataset of the DRIADA paper for the
[DANDI archive](https://dandiarchive.org) and turn a downloaded copy back into
the files the figure scripts read. The dataset is published as
[DANDI:002030](https://doi.org/10.48324/dandi.002030/0.261016.1321), version 0.261016.1321.

## What is in the dataset

39 sessions from 13 mice: three sessions per mouse, recorded on consecutive
days, about 12 minutes each, during free exploration of a circular open field.
One NWB file per session:

```
sub-F28/sub-F28_ses-FOF-F28-1D_behavior+ophys.nwb
sub-F28/sub-F28_ses-FOF-F28-2D_behavior+ophys.nwb
...
```

Each file holds

- `processing/ophys/Fluorescence/Calcium`: fluorescence traces (dF/F) of the
  retained CNMF-E components, frames x neurons, as they enter the analysis of
  the paper (negative values included);
- `processing/behavior`: position of the body centre (cm), speed (cm/s), body
  and head direction (radians), and the indicators of rest, walk, run
  (`locomotion`), freezing, rearing and of the wall, intermediate
  (`centermiddle`), central (`centertrue`) and non-wall (`center`) zones;
- subject (id, sex, date of birth, exact age on the day of the session), start
  of the session in Moscow time, device, indicator, virus and target
  coordinates.

All series share the time base of the imaging frames, nominal 30 frames per
second.

## What is not in the dataset

- Raw miniscope and behaviour videos.
- Spatial footprints of the neurons and their positions in the field of view.
  The ROI table has one row per neuron and a 1x1 placeholder mask.
- Results of the analysis (INTENSE tables); they are computed by
  `compute/run_intense.py`.

## From the archive to the figure scripts

```
pip install dandi
dandi download DANDI:002030/0.261016.1321
python publications/elife/dandi/nwb_to_synchronized.py <downloaded folder>
```

`nwb_to_synchronized.py` writes `synchronized/<session>_aligned.npz` into the
data root (`DRIADA_ELIFE_DATA`, see `../README.md`), or into `--out`. It needs
only `h5py` and `numpy`. The arrays it writes are identical to those of the
original files in order, type, shape and values.

## Scripts

| Script | What it does |
|---|---|
| `export_to_dandi.py` | packs `synchronized/*.npz` into NWB files in the layout of `dandi organize` |
| `create_dandiset.py` | creates the Dandiset with its metadata and writes `dandiset.yaml`; the API key is read from a file given by `--key` |
| `upload_dandiset.py` | uploads the NWB files to the draft and checks that the server holds exactly the local files with the same sizes |
| `validate_dandi.py` | NWB schema validation, NWB Inspector with the `dandi` configuration, `dandi validate`, dry run of `dandi organize` |
| `check_roundtrip.py` | compares every NWB file with its synchronized file, read directly and through `load_exp_from_nwb` |
| `nwb_to_synchronized.py` | rebuilds the `synchronized` layout from NWB files |
| `compare_synchronized.py` | compares rebuilt npz files with the originals array by array |
| `fof_metadata.py` | descriptive metadata written into the files |
| `subjects.csv`, `sessions.csv` | per-animal and per-session inputs of the exporter |

The folder of the NWB files is `--out`, else the environment variable
`DRIADA_ELIFE_DANDI`, else `<data root>/dandi_nwb`. Packing needs `pynwb`,
`h5py` and the `tools` folder of the library; checking also needs
`nwbinspector` and `dandi`.

## Reading a file with the library

`driada.experiment.nwb.load_exp_from_nwb` opens the files. It applies the
preprocessing of the library on load (negative values of the traces are
clipped to zero) and leaves out a behavioural variable that is constant over
the session. To get the arrays exactly as stored, use `nwb_to_synchronized.py`
or read the file with `pynwb` or `h5py`.
