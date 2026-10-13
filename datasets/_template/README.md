# Dataset template

Starting point for a new source. Copy the folder, then follow
[`docs/ADDING_A_DATASET.md`](../../docs/ADDING_A_DATASET.md):

```bash
cp -r datasets/_template datasets/<dataset_name>
```

| File | What to do |
|---|---|
| `config.py` | Fill in every field marked `TODO`: identity, paths, windowing, labels, metadata schema, license |
| `build_dataset.py` | Implement label inference and filename parsing for the source |
| `README.md` | Delete it after copying; document the dataset in `docs/DATASETS.md` instead |

Register the dataset in `datasets/registry.py`, then build it with
`python run.py <dataset_name>`.
