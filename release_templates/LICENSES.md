# RadioPile v1.0 — Per-File License Summary

License values below are taken from each source dataset's `LICENSE` constant
in the RadioPile pipeline configuration (`datasets/<name>/config.py`), which
records the license each source is distributed under. The evidence per source
is in `docs/DATASETS.md` of the pipeline repository.

| File | License | Commercial use | Share-alike |
|---|---|---|---|
| `aggregated/radiopile_v1.0.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `per_dataset/sub_ghz/sub_ghz.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `per_dataset/lte_wifi_dvbt/lte_wifi_dvbt.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `per_dataset/nist_wifi_bluetooth/nist_wifi_bluetooth.h5` | NIST-public-domain | Yes | No |
| `per_dataset/its_band/its_band.h5` | CC-BY-4.0 | Yes | No |
| `per_dataset/rml2016_10a/rml2016_10a.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `per_dataset/rml2018_01a/rml2018_01a.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `per_dataset/radchar/radchar.h5` | CC-BY-NC-SA-4.0 | No | Yes |

## Notes

* **Aggregated corpus license is the most restrictive of its sources.**
  Because the aggregated HDF5 mixes windows from datasets released under
  CC-BY-NC-SA-4.0 with windows from more permissive sources, the combined
  artifact inherits the strongest copyleft and non-commercial constraint
  present among its inputs. Five of the seven sources carry CC-BY-NC-SA-4.0:
  `sub_ghz`, `lte_wifi_dvbt`, `rml2016_10a`, `rml2018_01a` and `radchar`.
  Researchers needing a different licensing posture should use the
  appropriate per-dataset file standalone instead of the aggregated corpus.

* **Commercial-friendly subset.** The other two sources permit broader use:
  `nist_wifi_bluetooth` (NIST data terms) and `its_band`
  (CC-BY-4.0). Their per-dataset files were never mixed with other sources,
  so they are not bound by the aggregate's terms and may be used
  commercially. Both still require attribution.

* **Attribution.** All licenses in this bundle require attribution to the
  original source-dataset authors. Each file's `*.croissant.json` sidecar
  contains the canonical citation. The RadioPile paper additionally lists every
  source-dataset citation in its references section.

* **NIST data.** The NIST recordings are distributed under NIST's data terms
  (https://www.nist.gov/open/license), not under a Creative Commons license. Works of
  NIST employees are not subject to copyright protection in the United States
  (17 U.S.C. §105) and may be subject to foreign copyright; to the extent NIST holds
  copyright elsewhere, it grants a royalty-free worldwide right to use, modify and
  redistribute the data. That permission is contingent on acknowledging NIST as the
  source, and NIST asks that modified works carry a notice of the change. The NIST
  file in this release is such a modified work: it was processed by the RadioPile
  pipeline, as described in the README.

* **License authority.** The constants in `datasets/<name>/config.py` in
  the RadioPile source repository are the authoritative record. If the
  upstream source dataset's license changes after this bundle was built,
  the older license at build time still governs the bundled file under
  the terms recorded at build time.
