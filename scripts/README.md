# scripts/

Run everything from the repository root. Every script takes `--help`.

| Script | Purpose |
|---|---|
| `validate_datasets.py` | Validates the built HDF5 files; writes the validation report |
| `verify_raw_inputs.py` | Checks raw inputs against `docs/RAW_INPUT_CHECKSUMS.txt` |
| `audit_noise_filtering.py` | How many raw windows the noise filter drops, per source |
| `audit_aggregation_filtering.py` | Label routing and window-size adaptation, per source |
| `inspect_noise_floor.py` | Noise-floor inspection for one raw file (see `docs/NOISE_FILTERING.md`) |

**`analysis/`** — evidence behind decisions recorded in `docs/DATASETS.md`: the
Gentbrugge cell search, the NIST noise-filter checks, and the per-class source
breakdowns.

**`paper_figures/`** — scripts that write into `figures/`, the folder holding figures
and supporting figures for the paper.

**Release tooling (`_`-prefixed)** — maintainers only, in this order:

```bash
python scripts/_set_release_identifiers.py --record-a <doi> --record-b <doi> --code <doi>
python run.py aggregation
make release FORCE=1
python scripts/_stage_zenodo.py
python scripts/_zenodo_upload.py --deposition <id> --dir release/zenodo_upload/record_a
python scripts/_zenodo_upload.py --deposition <id> --dir release/zenodo_upload/record_b
```

`_zenodo_upload.py` needs `requests`, which is not in `requirements.txt`
(`pip install requests==2.34.2`).

The first command records the reserved DOIs in the source tree and is needed only
once. The build then writes them into the files, `make release` validates and
fingerprints the files under `data/` and copies them into the bundle, and staging
checks every staged HDF5 against that fingerprint. Never modify an HDF5 inside
`release/` after `make release`: staging will refuse it.

`_set_release_identifiers.py --patch-h5` writes the identifiers into an already
built file under `data/` without a rebuild, for the case where a DOI changes after
the build. The first command only fills placeholders, so a DOI that is already
recorded has to be changed by hand in `datasets/aggregation/config.py`, `README.md`,
`CITATION.cff` and `release_templates/` first. Run `--patch-h5` before `make release`. `_fix_bundle_croissant.py` is called by
`make release`. `_zenodo_upload.py` writes to an existing draft and cannot publish.
