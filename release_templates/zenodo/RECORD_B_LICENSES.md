# RadioPile v1.0 — Licenses (Per-Dataset Record)

**There is no single license for this record. Each file keeps the license of its
original source dataset.**

These files were processed individually and never mixed, so nothing is inherited
between them. This is the essential difference from the aggregated record
(10.5281/zenodo.22876704), where combining sources forces the whole artifact to the most
restrictive license present (CC-BY-NC-SA-4.0). Two of the seven files here are
therefore usable on terms the aggregate cannot offer.

License values are taken from each source dataset's `LICENSE` constant in the
RadioPile pipeline configuration (`datasets/<name>/config.py`), which records the
license each source is distributed under. The evidence per source is in
`docs/DATASETS.md` of the pipeline repository.

| File | License | Commercial use | Share-alike |
|---|---|---|---|
| `nist_wifi_bluetooth.h5` | NIST-public-domain | **Yes** | No |
| `its_band.h5` | CC-BY-4.0 | **Yes** | No |
| `sub_ghz.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `lte_wifi_dvbt.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `rml2016_10a.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `rml2018_01a.h5` | CC-BY-NC-SA-4.0 | No | Yes |
| `radchar.h5` | CC-BY-NC-SA-4.0 | No | Yes |

Each file's `.h5.config.json`, `.croissant.json` and `_config.py` sidecars carry the
same license as the `.h5` they describe.

`validation_report.json` is not covered by any of the above. It is output of the
RadioPile pipeline rather than a derivative of any source dataset — a record of
automated checks, containing no signal data — and is distributed under the
pipeline's Apache 2.0 license. The same file ships in both Zenodo records.

## Notes

* **Commercially usable subset.** Only `its_band` (CC-BY-4.0) and
  `nist_wifi_bluetooth` (NIST data terms) permit commercial use. Both
  still require attribution. Combining either with a CC-BY-NC-SA-4.0 file
  produces a derivative governed by the more restrictive terms — that is precisely
  what the aggregated record is.

* **Attribution.** Every set of terms here requires attribution to the
  original source-dataset authors. Each file's `*.croissant.json` contains the
  canonical citation. The RadioPile data descriptor additionally lists every
  source-dataset citation in its references.

* **NIST data.** The NIST recordings are distributed under NIST's data terms
  (https://www.nist.gov/open/license), not under a Creative Commons license. Works of
  NIST employees are not subject to copyright protection in the United States
  (17 U.S.C. §105) and may be subject to foreign copyright; to the extent NIST holds
  copyright elsewhere, it grants a royalty-free worldwide right to use, modify and
  redistribute the data. That permission is contingent on acknowledging NIST as the
  source, and NIST asks that modified works carry a notice of the change. The NIST
  file in this release is such a modified work: it was processed by the RadioPile
  pipeline, as described in the README.

* **Share-alike.** CC-BY-NC-SA-4.0 requires that derivatives be distributed under the
  same terms. This applies to a fine-tuned model's weights in some readings; seek your
  own advice if you intend to redistribute models trained on these files.

* **License authority.** The constants in `datasets/<name>/config.py` in the RadioPile
  source repository are the authoritative record. If an upstream license changes after
  this record was published, the license at build time still governs these files under
  the terms recorded at build time.
