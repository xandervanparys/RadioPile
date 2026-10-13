# RadioPile v1.0 — License (Aggregated Record)

License values below are taken from each source dataset's `LICENSE` constant
in the RadioPile pipeline configuration (`datasets/<name>/config.py`), which
records the license each source is distributed under. The evidence per source
is in `docs/DATASETS.md` of the pipeline repository.

| File | License | Commercial use | Share-alike |
|---|---|---|---|
| `radiopile_v1.0.h5` | CC-BY-NC-SA-4.0 | No | Yes |

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
  (CC-BY-4.0). Their per-dataset files in the per-dataset record
  (10.5281/zenodo.22877108) were never mixed with other sources, so they are not bound
  by the aggregate's terms and may be used commercially. Both still require
  attribution.

* **Attribution.** All licenses in this record require attribution to the
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
  upstream source dataset's license changes after this record was built,
  the older license at build time still governs the published file under
  the terms recorded at build time.


---

## This record contains only the aggregated file

The per-dataset files are published as a separate Zenodo record
(10.5281/zenodo.22877108), where **each file keeps its own original source license** rather
than the inherited CC-BY-NC-SA-4.0 above. Two of the seven — the NIST recordings
(NIST data terms) and ITS Band (CC-BY-4.0) — permit commercial use as standalone
files. If that matters for your use, take them from the per-dataset record.

`validation_report.json` is not covered by any of the above. It is output of the
RadioPile pipeline rather than a derivative of any source dataset — a record of
automated checks, containing no signal data — and is distributed under the
pipeline's Apache 2.0 license. The same file ships in both Zenodo records.
