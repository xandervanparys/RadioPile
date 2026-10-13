# Datasets

Everything that is true of one source: where it comes from, what its raw files look
like, how they are parsed, what was decided about it, and what to watch out for. This
is the file that grows when a dataset is added.

**Verification.** Values were checked against each dataset's `config.py` and
`build_dataset.py`, the raw inputs under `data/<dataset>/raw/` and the built files
under `data/<dataset>/processed/` (build of 2026-10-12). Values that come only from
upstream papers or READMEs, and so cannot be checked against code or files, are marked
**†**. When the code and this document disagree, the code is right.

Sections: [Rebuilding](#rebuilding-from-the-original-sources) ·
[`sub_ghz`](#sub_ghz--sub-ghz-technology-recognition-fontaine-et-al) ·
[`lte_wifi_dvbt`](#lte_wifi_dvbt--lte--wifi--dvb-t-technology-recognition-fontaine-et-al) ·
[`nist_wifi_bluetooth`](#nist_wifi_bluetooth--nist-wifi-iq-recordings-tschimben-et-al) ·
[`its_band`](#its_band--its-band-technology-recognition-girmay-et-al) ·
[`rml2016_10a`](#rml2016_10a--radioml-201610a-oshea--west) ·
[`rml2018_01a`](#rml2018_01a--radioml-201801a-oshea-et-al) ·
[`radchar`](#radchar--radchar-radar-signal-characterisation-huang-et-al) ·
[Classes shared across sources](#classes-shared-across-sources) ·
[Raw sample count](#raw-sample-count)

Each section lists the fields of the built file. Those are the dataset's
`WINDOW_METADATA_SCHEMA` plus `file_path`, which the writer adds to every file; the
schema itself, the fill values and the HDF5 layout are in `DESIGN.md` §3. Each
section also gives the source's **group key** for train/test splitting; why a key is
needed and how to split the aggregated file are in `DESIGN.md` §4.

---

## Rebuilding from the original sources

To rebuild, fetch each source and place it under `data/<dataset>/raw/`:

1. **Where to download:** the per-source sections below. Where a source ships more
   than the pipeline uses, the section says which files are built.
2. **Where files go:** [`RAW_INPUT_CHECKSUMS.txt`](RAW_INPUT_CHECKSUMS.txt) lists the
   exact path of every raw input. `sub_ghz` and `lte_wifi_dvbt` scan `raw/`
   recursively; the other five read their files directly under `raw/`. Filenames are
   parsed for capture parameters and must be kept.
3. **Whether you have the right bytes:** two sources are not versioned upstream, so
   check before building:

   ```bash
   python scripts/verify_raw_inputs.py                       # everything
   python scripts/verify_raw_inputs.py --dataset sub_ghz     # one source
   python scripts/verify_raw_inputs.py --data-root /mnt/raw  # data kept elsewhere
   ```

Then `python run.py <dataset>` per source, or `python run.py aggregation` for the whole
corpus (it rebuilds every source first). Validation steps are in
[`ADDING_A_DATASET.md`](ADDING_A_DATASET.md).

---

## sub_ghz — Sub-GHz Technology Recognition (Fontaine et al.)

**Citation:** J. Fontaine, A. Shahid, R. Elsas, A. Seferagic, I. Moerman, E. De Poorter,
"Multi-band sub-GHz technology recognition on NVIDIA's Jetson Nano," *IEEE 92nd VTC-Fall*,
2020.

**Repository:** https://github.com/JaronFontaine/Sub-GHz-IQ-signals-dataset
**Data files:** https://cloud.ilabt.imec.be/index.php/s/bqXtdp9QsfXLbb3

**License:** CC BY-NC-SA 4.0 (`CC-BY-NC-SA-4.0` in the config), stated in the "License"
section of the repository README. The repository has no separate license file.

Real over-the-air captures of five sub-GHz IoT technologies plus noise.

| | |
|---|---|
| Raw format | MATLAB 5.0 `.mat` (read with `scipy.io.loadmat`), 125 files under `raw/<band>/<technology>/` |
| Variable | `IQ_samples`, 1-D complex128 |
| Hardware / rate | RTL-SDR †, 2.048 Msps |
| Center frequencies | 864.0 and 867.4 MHz, pooled in one dataset; parsed from the filename and stored per window, not as a label |
| Raw labels (6) | `sigfox`, `lora`, `80211ah`, `802154g`, `sunofdm`, `noise` |
| Classes (5) | `WiFi`, `IEEE 802.15.4`, `LoRa`, `Noise`, `Sigfox` |
| Windows | 927,152 × 500 samples, hop 500; 2,048,000 before noise filtering |
| Noise filtering | `"auto"`: 1st percentile + 6 dB, per file; `noise` files are never filtered |
| Aggregation | none; already 500 samples |
| Group key | `file_path` + `recording_id` |

**Filename convention.**
`<tech>[_mcs<N>][_chan<N>]_g<gain>dB_att<att>dB_freq<freq>MHz_<run>.mat`, for example
`80211ah_mcs0_chan1_g0.0dB_att10dB_freq864.0MHz_0.mat`. Each token is matched
independently by `parse_metadata_filename()`, so tokens a technology does not use are
simply absent. LoRa files carry `sf<N>` for the spreading factor; only SF7 and SF12
occur.

**Six captured technologies map to five labels.** The variant moves into a metadata
field so the label space stays comparable with the other technology sources (the rule
is in `DESIGN.md` §1.6):

| Raw label | Class | Variant field |
|---|---|---|
| `sigfox` | `Sigfox` | |
| `lora` | `LoRa` | `spreading_factor` (7 or 12) |
| `80211ah` | `WiFi` | `wifi_standard = "802.11ah"` |
| `802154g` | `IEEE 802.15.4` | `ieee802154_variant = "g"` |
| `sunofdm` | `IEEE 802.15.4` | `ieee802154_variant = "SUN-OFDM"` (the field is the only way to tell the two apart) |
| `noise` | `Noise` | |

**Modulation and waveform assignment.** `MCS_TO_MODULATION` is one flat dictionary for
both OFDM technologies. That is safe only because their MCS sets do not overlap:
802.11ah uses {0, 7, 10}, SUN-OFDM uses {2, 6}.

| Raw label | `modulation_type` | `waveform_type` |
|---|---|---|
| `sigfox` | `BPSK` | `PSK` |
| `lora` | `CSS` | `CSS` |
| `80211ah`, MCS 0 or 10 | `BPSK` | `OFDM` |
| `80211ah`, MCS 7 | `64QAM` | `OFDM` |
| `802154g` | `BFSK` | `FSK` |
| `sunofdm`, MCS 2 | `OQPSK` | `OFDM` |
| `sunofdm`, MCS 6 | `16QAM` | `OFDM` |
| `noise` | `unknown` | `noise` |

Every OFDM file carries an MCS token, so `modulation_type` is set for 91.2% of windows
and `Noise` is the only class left `"unknown"` (81,920 windows).

**Metadata fields (14).** `technology_type`, `modulation_type`, `waveform_type`,
`center_frequency_hz`, `sample_rate_hz`, `gain_db`, `attenuation_db`, `mcs`, `channel`,
`recording_id`, `spreading_factor`, `wifi_standard`, `ieee802154_variant`, `file_path`.
`gain_db` is 0.0 for every window. The constant sample rate is written by declaring
2,048,000 as the fill of `sample_rate_hz`.

**Splitting.** `recording_id` is the run number at the end of the filename (`"0"` to
`"9"`) and repeats in every file, so on its own it is not a group key. One group is
one raw capture file: `file_path` + `recording_id`, 125 groups.

---

## lte_wifi_dvbt — LTE / WiFi / DVB-T Technology Recognition (Fontaine et al.)

**Citations:**
- J. Fontaine, E. Fonseca, A. Shahid, M. Kist, L. A. DaSilva, I. Moerman, E. De Poorter, "Towards low-complexity wireless technology classification across multiple environments," *Ad Hoc Networks*, vol. 91, p. 101881, 2019. https://doi.org/10.1016/j.adhoc.2019.101881
- W. Liu, M. Kulin, T. Kazaz, A. Shahid, I. Moerman, E. De Poorter, "Wireless technology recognition based on RSSI distribution at sub-Nyquist sampling rate for constrained devices," *Sensors*, vol. 17, no. 9, p. 2081, 2017. https://doi.org/10.3390/s17092081

**Repository:** https://github.com/JaronFontaine/Technology-Recognition-dataset-of-real-life-LTE-WiFi-and-DVB-T
**Data files:** https://cloud.ilabt.imec.be/index.php/s/qrJCWgzQaGPfHPr

**License:** CC BY-NC-SA 4.0 (`CC-BY-NC-SA-4.0` in the config), stated in the "License"
section of the repository README. The repository has no separate license file.

Real over-the-air captures of LTE, WiFi and DVB-T at six sites in and around Ghent,
Belgium.

| | |
|---|---|
| Raw format | GNU Radio `.bin`, complex64 (float32 I, float32 Q, 8 bytes per sample), 190 files under `raw/<location>/` |
| Hardware / rate | USRP, model not named upstream †; 1 Msps, or 10 Msps for files with a `10Msps` token |
| Center frequencies | 482, 806, 2412, 5180 and 5240 MHz; parsed from the filename and stored per window |
| Capture sites (6) | `igent`, `rabot`, `reep`, `uz`, `merelbeke`, `gentbrugge` |
| Raw labels (3) | `lte`, `wf`, `dvbt` |
| Classes (3) | `LTE`, `WiFi`, `DVB-T` |
| Windows | 415,344 × 500 samples, hop 500, last window of a file zero-padded |
| Noise filtering | none; signals are continuous |
| Aggregation | none; already 500 samples |
| Group key | `file_path` + `recording_id` |

All three technologies are OFDM, so `waveform_type = "OFDM"` for every window. The
subcarrier modulation cannot be resolved without MCS metadata, so
`modulation_type = "unknown"` for every window.

**Both papers are cited, one is the source.** The upstream README asks users to refer
to both publications, so both stay in the citation list here, in `config.py` and in the
Croissant descriptors. Fontaine et al. 2019 is the originating publication of these
captures. Liu et al. 2017 describes a separate campaign (WiFi at 5540 MHz, an Anritsu
analyzer, one office building †) and contributed no files: cite it, do not describe it
as a source.

**Trap: the `.bin` files are complex64, not int16.** Read as int16, the data looks
like noise rather than like a bug: a near-zero I channel with occasional spikes beyond
the expected range and a Q channel that dominates, because float32 exponent and
mantissa bytes are being read as separate integers. The config sets
`SOURCE_DTYPE = "complex64"`. The general rule is in `DESIGN.md` §2.3.

### RF capture parameters

Not stored in any config: parsed from the filename
`<tech>[<N>Msps]_g<gain>_<location>_f<freq>MHz_r<run>.bin` by
`parse_lte_wifi_dvbt_filename()`. The class comes from the leading alphabetic token.

| Technology | Center frequency | Band † | Files at 1 Msps | Files at 10 Msps |
|---|---|---|---|---|
| DVB-T | 482 MHz | UHF channel 22 | 60 | 5 |
| LTE | 806 MHz | 3GPP Band 20 downlink | 60 | 12 |
| WiFi | 2412 MHz | 2.4 GHz channel 1 | 11 | none |
| WiFi | 5180 MHz | 5 GHz channel 36 | 3 | 1 |
| WiFi | 5240 MHz | 5 GHz channel 48 | 30 | 8 |

Without an `<N>Msps` token the rate is 1 Msps, which the upstream README states as the
sampling rate used †. Complex sampling means the analyzed bandwidth equals the sample
rate.

### Misnamed Gentbrugge capture

`raw/gentbrugge/lte10Msps_g30_uz_f806MHz_r1.bin` is labeled `location = "gentbrugge"`
although its filename says `uz`. It arrived from upstream this way; for the other 189
files folder and filename agree. Built as named, its 2,200 windows would carry the
wrong location.

The signal settles it. Every LTE cell broadcasts a physical cell ID (PCI) in its
synchronization signals, within the central 0.93 MHz of the carrier, so PCIs are
decodable from every 806 MHz capture regardless of sample rate. Across the sixty 1 Msps
captures no PCI is heard at more than one site, so each site's PCI set is a location
fingerprint. The disputed file hears all four Gentbrugge cells (PCIs 72, 61, 297, 159,
each ≥ 8.6 standard deviations above the other candidates) and none of UZ's three
(286, 29, 25). Of the eleven correctly named 10 Msps captures, ten hear only their
own site's cells and one (`igent`, gain 20) yields no detection at all, which is the
control. Supporting detail: the file is not a copy of the UZ
capture of the same name (cross-correlation peak 0.003), its upstream timestamp falls
inside the Gentbrugge recording slot, and its power matches Gentbrugge's 1 Msps
captures. Reproduce with `python scripts/analysis/lte_cell_search.py`.

The fix is a one-file `LOCATION_OVERRIDES` entry in `datasets/lte_wifi_dvbt/config.py`,
covered by `tests/unit/test_location_override.py`. The raw file is not renamed, so
the copy stays identical to upstream, and the reader still warns if any other folder
and filename disagree. Result: gentbrugge 70,400 windows, uz 57,200; IQ data, labels
and all other fields unchanged; `file_path` still shows the upstream name. `location`
is not carried into the aggregate.

**Metadata fields (10).** `technology_type`, `modulation_type`, `waveform_type`,
`wifi_standard`, `sample_rate_hz`, `center_frequency_hz`, `gain_db`, `location`,
`recording_id`, `file_path`. `wifi_standard` is `"unknown"` for WiFi windows, because
the 802.11 variant is not documented upstream, and `""` for LTE and DVB-T windows.

**Splitting.** `recording_id` is the run number (`"1"` to `"10"`) and repeats in
every file, so on its own it is not a group key. One group is one raw capture file:
`file_path` + `recording_id`, 190 groups.

---

## nist_wifi_bluetooth — NIST WiFi IQ Recordings (Tschimben et al.)

**Citations:**
- S. Tschimben, S. Subray, A. Sanders, A. Wunderlich, "Wi-Fi and Bluetooth I/Q Recordings in the 2.4 GHz and 5 GHz Bands with Low-Cost Software Defined Radios," NIST data repository, 2022. https://doi.org/10.18434/mds2-2731
- Same authors and title, NIST Technical Note 2237, 2022. https://doi.org/10.6028/NIST.TN.2237

**License:** NIST data terms, not a Creative Commons license; `NIST-public-domain` in
the config is RadioPile's own label for them. The NIST record (version 1.3.2) gives
https://www.nist.gov/open/license as its license, and item 7 of the dataset's
`README.txt` lists no other license or restriction. Under those terms, works of NIST
employees are not subject to copyright in the United States (17 U.S.C. §105) and may
be subject to foreign copyright; NIST grants a royalty-free worldwide right to use,
modify and redistribute, contingent on acknowledging NIST, and asks that modified
works carry a notice of the change. Two of the four authors are listed with the
University of Colorado Boulder; NIST's record applies the same terms to the whole
dataset.

Real over-the-air WiFi captures at the University of Colorado Boulder, November
2021 †: indoor and outdoor, 2.4 and 5 GHz. All windows are WiFi; the dataset
contributes scale, channel and environment diversity rather than class diversity.

| | |
|---|---|
| Raw format | HDF5, one file per capture condition, 180 captures per file |
| Capture layout | each capture is one dataset of 60,000,000 int16 values: interleaved I/Q, 1 s at 30 Msps |
| Hardware / rate | USRP B200-mini-i †, 30 Msps |
| Center frequencies | 2.437 GHz and 5.825 GHz, from `FILE_CONFIG` |
| Classes (1) | `WiFi` |
| Windows | 20,627,827 × 500 samples (16.7 µs at 30 Msps), hop 500, no padding |
| Noise filtering | `"auto"`: 1st percentile + 6 dB, detected once per file from its first capture |
| Aggregation | none; already 500 samples |
| Group key | `file_path` + `recording_id` |

### Files

NIST publishes five files; three are built.

| File | Center | Environment | Standard | MCS † | Modulation | Windows kept | Decision |
|---|---|---|---|---|---|---|---|
| `2_4ghz_indoor.h5` | 2.437 GHz | indoor | 802.11n | 15 | 64QAM | 9,559,474 (88.5%) | built |
| `5ghz_indoor.h5` | 5.825 GHz | indoor | 802.11ac | 8 | 256QAM | 7,407,438 (68.6%) | built |
| `5ghz_outdoor.h5` | 5.825 GHz | outdoor | 802.11ac | 8 | 256QAM | 3,660,915 (33.9%) | built; the filter also drops lower-power WiFi-like activity, see below |
| `2_4ghz_outdoor.h5` | 2.437 GHz | | | | | | excluded: occasional Bluetooth FHSS alongside WiFi, so windows cannot be labeled WiFi-only |
| `2_4ghz_bluetooth.h5` | 2.437 GHz | | | | | | excluded: intentional WiFi + Bluetooth co-channel captures |

Percentages are of the 10.8 million windows a file holds before filtering (180
captures × 60,000). The two excluded files are candidates for an unlabeled
pre-training mode or multi-label windows.

**Loading.** `int16 / 32768.0` gives float32 in [−1, 1], reshaped to `(-1, 2)`. The
noise threshold is computed from the first capture only, which avoids loading all 180
captures to detect it.

**Modulation and standard are per-file constants.** MCS was monitored on the iPerf
client's link once per second (`iw dev wlan0 link`) and was stable for each session
(TN 2237 §3.4 †), so `modulation_type` and `wifi_standard` come from `FILE_METADATA`
in `config.py`, not from decoding frames. Traffic ran iPerf client → access point →
iPerf server; the note reports monitoring on the client link only, so the annotation
is not verified for the access point's own transmissions. This is the only
real-capture source in which `modulation_type` is set for every window. The NIST
landing page and README say nothing about MCS; it is the companion technical note that
documents it (`DESIGN.md` §2.3).

### Noise filtering

WiFi is packet-based, so a large share of windows fall in inter-frame gaps. Each file's
noise floor is the 1st percentile of window power in its first capture, plus a 6 dB
margin; windows below that are dropped. This was checked against a simple amplitude
threshold (`max(|I|) > 0.05`) on 20 captures per file
(`scripts/analysis/verify_nist_noise_filtering.py`):

| File | Noise floor | Threshold | Agreement with amplitude method |
|---|---|---|---|
| `2_4ghz_indoor` | −61.0 dB | −55.0 dB | 96.5% |
| `5ghz_indoor` | −57.8 dB | −51.8 dB | 90.0% |
| `5ghz_outdoor` | −41.5 dB | −35.5 dB | 98.1% |

The two methods never disagree in the other direction on the indoor files: every
window the amplitude method keeps, the power method keeps too. The power method keeps
more (88.9% against 85.4% at 2.4 GHz, 68.6% against 58.5% at 5 GHz), because the
amplitude method checks only the I channel and misses windows whose signal is in Q. Indoor captures have a wide gap
between floor and signal and tolerate a wide range of margins.

`5ghz_outdoor` is the hard case. Its floor is about 16 dB above that of `5ghz_indoor`,
consistent with its higher receive gain (65 against 30 in the per-file metadata CSVs;
TN 2237 Table 8 gives a matching difference in the receiver's own noise floor †), which
leaves only about 13 dB between floor and peak. Between the floor and the main signal
mode this file has a third mode at −38 to −36 dB (about 22% of windows in the
20-capture sample) whose activity is consistent with WiFi: the same 20 MHz band and
center as the main mode, at lower power.

**Margin sweep on `5ghz_outdoor`.** One run over the same 20 captures (1.2 million
windows) gives two different quantities per margin. *Retention* is the share of
windows the power threshold keeps. *Agreement* is the share of windows on which the
power threshold and the amplitude reference make the same keep-or-drop decision. The
amplitude reference itself keeps 32.5% of these windows.

| Margin | Threshold | Retention | Agreement |
|---|---|---|---|
| 3 dB | −38.5 dB | 57.4% | 75.0% |
| 6 dB | −35.5 dB | 34.0% | 98.1% |
| 10 dB | −31.5 dB | 12.1% | 79.3% |

A 3 dB margin keeps the third mode; 10 dB cuts into the main signal mode; 6 dB falls
between the two, so the released windows are the main mode only. The sweep is produced
by `scripts/paper_figures/noise_filtering_validation.py`. Spectrograms of kept and
dropped windows (`scripts/analysis/verify_nist_sample_quality.py`) were inspected for
all three files. The general lesson is in `DESIGN.md` §2.6.

### Known limitations

- `5ghz_outdoor` dropped windows are not only idle: they include lower-power activity
  consistent with WiFi, so the kept windows under-represent the WiFi traffic in this
  file.
- No per-window SNR is provided.
- `modulation_type` is a session constant taken from MCS monitoring on the iPerf
  client's link; it is not verified for other transmitters in the capture, and
  individual frames may transiently use another MCS during rate adaptation.
- The noise floor comes from the first capture of each file only. Stable in practice,
  since the floor is thermal or environmental rather than signal-dependent.

**Metadata fields (10).** `technology_type`, `modulation_type`, `waveform_type`,
`wifi_standard`, `center_frequency_hz`, `sample_rate_hz`, `environment`,
`recording_id`, `sample_index`, `file_path`. `sample_index` is the window's 0-based
position within its capture. The schema declares `"unknown"` as the fill for
`recording_id` and `environment`; both are set for every window, so it never appears.

**Splitting.** `recording_id` is the capture's key in the NIST HDF5 file, for example
`"144700"`, and `file_path` names the HDF5 file it came from; together they give 540
groups, in this file and in the aggregate. The 5 GHz indoor and outdoor files were
recorded simultaneously and share capture keys. A split that keeps those simultaneous
captures on one side groups on `recording_id` alone (360 groups, the 2.4 GHz keys
being distinct); the paper's baselines do this.

---

## its_band — ITS Band Technology Recognition (Girmay et al.)

**Citation:** M. Girmay, V. Maglogiannis, D. Naudts, M. Aslam, A. Shahid, I. Moerman,
"Technology recognition and traffic characterization for wireless technologies in ITS
band," *Vehicular Communications*, vol. 39, p. 100563, 2023.
https://doi.org/10.1016/j.vehcom.2022.100563

**Repository:** https://gitlab.ilabt.imec.be/mgirmay/tech-rec-its-band

**License:** CC BY 4.0 (`CC-BY-4.0` in the config), confirmed in writing by the dataset
authors. The repository itself carries no license statement, so that confirmation is
the evidence for the six files used here. The IEEE Dataport record
(https://doi.org/10.21227/72qq-z464) lists CC BY 4.0 in its DOI metadata, but it holds
the compiled file that was not used.

The corpus's only V2X source: ITS-G5 and C-V2X PC5 alongside LTE, 5G NR, WiFi and
noise, captured on the imec Smart Highway testbed (E313 near Antwerp) †. The released
data has synthetic AWGN added to the over-the-air captures at SNRs
{−10, −5, …, 30} dB for all six classes (paper §4.2 †), so it is a hybrid, not a purely
real dataset. The SNR of an observation is not recorded in the files, so there is no
`snr_db` field. RadioPile stores the raw IQ as released; the paper itself trained on
FFTs.

| | |
|---|---|
| Raw format | MATLAB v7.3 `.mat` (HDF5), six files, float64 |
| Sample rate | 25 Msps, the highest of the six rates the source offers (1, 5, 10, 15, 20, 25 Msps †) |
| Center frequency | 5.9 GHz, a constant in the config † |
| Hardware | USRP X310 with srsRAN (LTE), OpenAirInterface (5G NR), Cohda MK5 (ITS-G5), Cohda MK6c (C-V2X PC5), openwifi on Xilinx Zynq (WiFi) † |
| Classes (6) | `LTE`, `5G NR`, `ITS-G5`, `C-V2X PC5`, `WiFi`, `Noise` |
| Windows | 90,000 × 500 samples: 6 classes × 7,500 observations × 2 windows |
| Noise filtering | none; signals are continuous |
| Aggregation | none; observations are already cut to 500 samples in this source's own build |
| Group key | `recording_id` |

**Files used.** The six per-technology 25 Msps files. Each holds one variable. Read
with h5py it has shape `(2200, 7500)`; the build transposes it to `(7500, 2200)`, one
row per observation. The first 1100 values of a row are I, the last 1100 are Q.

| File | Variable | Class | `waveform_type` |
|---|---|---|---|
| `LTE_25msps.mat` | `LTE` | `LTE` | `OFDM` |
| `FiveG_25msps.mat` | `Five_G` | `5G NR` | `OFDM` |
| `ITSG5_25msps.mat` | `ITSG5` | `ITS-G5` | `OFDM` (802.11p-based) |
| `CV2X_25msps.mat` | `CV2X` | `C-V2X PC5` | `SC-FDM` |
| `WiFi_25msps.mat` | `WiFi` | `WiFi` | `OFDM` |
| `noise_25msps.mat` | `Noise` | `Noise` | `noise` |

**Not used.** The compiled `dataset_noise_added.mat` from IEEE Dataport (undocumented
augmentation, 18,500 vs 7,500 samples per class, only five classes †) and the `new/`
and `*_validat*` folders (undocumented conditions †). `raw/previous_raw_files/` holds
the lower-rate exploration files and is not read by the build.

**Observation splitting.** An observation is 1100 samples (44 µs). It is cut into two
500-sample windows and the last 100 samples (9%) are discarded, rather than cropped to
one window, which would discard 55%. `sample_index` holds the observation index (0 to
7,499).

**Decisions.**
- 25 Msps is the highest of the six available rates. Downstream users can decimate;
  upsampling a lower rate cannot recover detail. The 10 Msps files would need only
  12% padding for their 440-sample observations, but lose that detail (`DESIGN.md`
  §2.4).
- WiFi is 802.11n, MCS 0–7, from an openwifi SDR (paper §4.1.1 and the repository
  README †). `wifi_standard = "802.11n"` for WiFi windows and `""` otherwise;
  `modulation_type = "unknown"` for every window, because MCS varies per observation
  and is not recorded per sample.
- **The WiFi class comes from `WiFi_25msps.mat` only.** That file matches its siblings
  (shape `(2200, 7500)` float64, expected variable name). The upstream repository also
  holds 17 WiFi `.bin` files with no sampling-rate metadata, sitting next to 10 Msps
  `.mat` files; folder placement is not evidence of sampling rate, so they are not
  used. The principle is in `DESIGN.md` §2.3.
- The journal paper is cited rather than the IEEE Dataport entry, because the Dataport
  compilation is not what was used.

**Metadata fields (9).** `technology_type`, `modulation_type`, `waveform_type`,
`wifi_standard`, `center_frequency_hz`, `sample_rate_hz`, `sample_index`,
`recording_id`, `file_path`.

**Splitting.** `recording_id = "<file stem>_<observation index>"`, for example
`"LTE_25msps_42"`, links the two windows cut from one observation and is unique across
the dataset: 45,000 groups of two. Each class comes from a single source file, so
grouping keeps an observation whole but cannot hold out a recording.

---

## rml2016_10a — RadioML 2016.10A (O'Shea & West)

**Citations:**
- T. J. O'Shea, N. West, "Radio Machine Learning Dataset Generation with GNU Radio," *Proc. 6th GNU Radio Conference*, 2016.
- T. J. O'Shea, J. Corgan, T. C. Clancy, "Convolutional Radio Modulation Recognition Networks," *EANN*, 2016.
- Dataset: DeepSig Inc., RadioML 2016.10A. https://www.deepsig.ai/datasets
- Mirror: https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom

**Available from:** the dataset is published by DeepSig (https://www.deepsig.ai/datasets),
whose direct download link
(https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2) did not serve the
archive as of October 2026. A mirror is at
https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom; its
`RML2016.10a_dict.pkl` is byte-identical to the file used for v1.0 (same SHA-256 as in
[`RAW_INPUT_CHECKSUMS.txt`](RAW_INPUT_CHECKSUMS.txt)). Check any download with
`python scripts/verify_raw_inputs.py --dataset rml2016_10a`.

**License:** CC BY-NC-SA 4.0 (`CC-BY-NC-SA-4.0` in the config). DeepSig's dataset page
(https://www.deepsig.ai/datasets) states that all datasets it provides, RadioML
2016.10A among them, are under this license. DeepSig's notice is the license evidence,
not the mirror.

Synthetic modulation dataset generated in GNU Radio with LO drift and light fading †.

| | |
|---|---|
| Raw format | one Python pickle, `raw/RML2016.10a_dict.pkl` |
| Structure | dict keyed `(modulation, snr)` mapping to float32 arrays of shape `(1000, 2, 128)` |
| Sample rate | none; simulated |
| SNR | −20 to +18 dB in 2 dB steps, 20 levels, stored as `snr_db` |
| Classes (11) | `BPSK`, `QPSK`, `8PSK`, `PAM4`, `16QAM`, `64QAM`, `GFSK`, `CPFSK`, `WBFM`, `AM-DSB`, `AM-SSB` |
| Windows | 220,000 × 128 samples: 11 modulations × 20 SNRs × 1000; each entry is already a window |
| Noise filtering | none; SNR is part of the design |
| Aggregation | zero-padded from 128 to 500 samples (372 zeros, 74% of the window) |
| Group key | none; simulated i.i.d. |

**Trap: the pickle holds Python 2 byte strings.** The file must be loaded with
`pickle.load(f, encoding="bytes")`, and the dictionary keys then arrive as `bytes`
(`b'BPSK'`). `load_rml2016_10a_data()` decodes them before use. `LABEL_REMAP` also
maps the upstream spellings `QAM16` and `QAM64` to `16QAM` and `64QAM`.

**Waveform family.**

| Modulation | `waveform_type` |
|---|---|
| `BPSK`, `QPSK`, `8PSK` | `PSK` |
| `16QAM`, `64QAM` | `QAM` |
| `GFSK`, `CPFSK` | `FSK` |
| `PAM4` | `PAM` |
| `WBFM` | `FM` |
| `AM-DSB`, `AM-SSB` | `AM` |

**The only source below the 500-sample target.** Every window is zero-padded at
aggregation and, by design, not renormalized afterward (`DESIGN.md` §2.4). In the
aggregate its windows therefore measure a mean power of 0.256 (128 / 500) over all 500
samples and 1.000 over the 128 real ones.

**Unit-power normalization does not remove the SNR information.** SNR is
scale-invariant (`DESIGN.md` §2.5). What remains is whether absolute power was an extra
cue. Measured on the raw pickle, mean per-example power against SNR:

| SNR (dB) | −20 | −10 | −4 | 0 | +4 | +10 | +18 |
|---|---|---|---|---|---|---|---|
| mean power (×10⁻⁵) | 7.82 | 7.78 | 7.61 | 7.19 | 6.89 | 6.84 | 6.86 |

Total power varies 13% end to end, is negatively correlated with SNR (Pearson
r = −0.46 over examples), and is flat below −10 dB and above +4 dB. The dataset was
synthesized with total power roughly constant, so absolute scale says almost nothing
about SNR, and what it says points the wrong way. A model that learned "lower power ⇒
higher SNR" would be learning the generator, not radios. Within the dataset the
operation is nearly a no-op anyway: the normalization divisor has a coefficient of
variation of 5.3%. `snr_db` remains available for SNR-stratified evaluation.

**Metadata fields (5).** `modulation_type`, `waveform_type`, `snr_db`, `sample_index`,
`file_path`. `sample_index` (0 to 999) is the position within the `(modulation, snr)`
group.

**Splitting.** No group key: examples are simulated independently, there is no
`recording_id` field in this file, and the aggregate fills it with `""`. A random split
is safe.

---

## rml2018_01a — RadioML 2018.01A (O'Shea et al.)

**Citations:**
- T. J. O'Shea, T. Roy, T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," *IEEE JSTSP*, vol. 12, no. 1, pp. 168–179, 2018.
- Dataset: DeepSig Inc., RadioML 2018.01A. https://www.deepsig.ai/datasets
- Mirror: https://www.kaggle.com/datasets/pinxau1000/radioml2018

**Available from:** the dataset is published by DeepSig (https://www.deepsig.ai/datasets),
whose direct download link
(https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz) did
not serve the archive as of October 2026. A mirror is at
https://www.kaggle.com/datasets/pinxau1000/radioml2018; its
`GOLD_XYZ_OSC.0001_1024.hdf5`, `LICENSE.TXT` and `classes-fixed.txt` are byte-identical
to the files used for v1.0 (same SHA-256 as in
[`RAW_INPUT_CHECKSUMS.txt`](RAW_INPUT_CHECKSUMS.txt)). Check any download with
`python scripts/verify_raw_inputs.py --dataset rml2018_01a`.

**License:** CC BY-NC-SA 4.0 (`CC-BY-NC-SA-4.0` in the config), per the `LICENSE.TXT`
distributed with the dataset archive (kept at `raw/LICENSE.TXT`, copyright DeepSig
Inc.) and the license notice on the DeepSig dataset page.

The larger successor to RML2016.10A.

| | |
|---|---|
| Raw format | one HDF5 file, `GOLD_XYZ_OSC.0001_1024.hdf5`, 21.4 GB |
| Structure | `X`: (2,555,904, 1024, 2) float32; `Y`: (2,555,904, 24) int64 one-hot; `Z`: (2,555,904, 1) int64 SNR |
| Sample rate | none; simulated |
| SNR | −20 to +30 dB in 2 dB steps, 26 levels, stored as `snr_db` |
| Classes (24) | `OOK`, `4ASK`, `8ASK`, `BPSK`, `QPSK`, `8PSK`, `16PSK`, `32PSK`, `16APSK`, `32APSK`, `64APSK`, `128APSK`, `16QAM`, `32QAM`, `64QAM`, `128QAM`, `256QAM`, `AM-SSB-WC`, `AM-SSB-SC`, `AM-DSB-WC`, `AM-DSB-SC`, `FM`, `GMSK`, `OQPSK` |
| Windows | 2,555,904 × 1024 samples: 24 modulations × 26 SNRs × 4096; each row is already a window |
| Noise filtering | none; SNR is part of the design |
| Aggregation | start-cropped from 1024 to 500 samples (524 dropped) |
| Group key | none; simulated i.i.d. |

**Trap: the one-hot label indices are scrambled.** The class order distributed with
the dataset does not match the indices in `Y`. `raw/classes-fixed.txt` gives the
corrected order, and `MODULATION_TYPES` in the config reproduces it: index `i` of the
one-hot vector is `MODULATION_TYPES[i]`, in the order of the Classes row above. Using
the original order (which starts `32PSK`, `16APSK`, `32QAM`, …) mislabels every class.

**Waveform family.**

| Modulation | `waveform_type` |
|---|---|
| `OOK`, `4ASK`, `8ASK` | `ASK` |
| `BPSK`, `QPSK`, `8PSK`, `16PSK`, `32PSK`, `OQPSK` | `PSK` |
| `16APSK`, `32APSK`, `64APSK`, `128APSK` | `APSK` |
| `16QAM`, `32QAM`, `64QAM`, `128QAM`, `256QAM` | `QAM` |
| `AM-SSB-WC`, `AM-SSB-SC`, `AM-DSB-WC`, `AM-DSB-SC` | `AM` |
| `FM` | `FM` |
| `GMSK` | `FSK` |

**Reading.** The file is read 4096 rows at a time (about 32 MiB) so it never has to be
loaded whole. Each row is transposed from `(1024, 2)` to `(2, 1024)`.

Five classes overlap with RML2016.10A (`BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`),
which is what allows cross-source comparison of one modulation
(`scripts/analysis/visualize_class_across_sources.py`).

**Metadata fields (5).** `modulation_type`, `waveform_type`, `snr_db`, `sample_index`,
`file_path`. `sample_index` is the row number in the raw file (0 to 2,555,903).

**Splitting.** No group key: examples are simulated independently, there is no
`recording_id` field in this file, and the aggregate fills it with `""`. A random split
is safe.

---

## radchar — RadChar Radar Signal Characterisation (Huang et al.)

**Citation:** Z. Huang, A. Pemasiri, S. Denman, C. Fookes, T. Martin, "Multi-Task Learning
For Radar Signal Characterisation," *IEEE ICASSPW*, 2023.
https://doi.org/10.1109/ICASSPW59220.2023.10193318

**Repository:** https://github.com/abcxyzi/RadChar
**Data files:** https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023

**License:** recorded as CC BY-NC-SA 4.0 (`CC-BY-NC-SA-4.0` in the config), the most
restrictive of the designations the author gives. The Kaggle page, which is the
official download, lists CC BY-NC-SA 4.0. The repository README badge is labeled "CC
BY-NC 4.0" but links to the BY-NC-SA 4.0 deed, and the QUT Research Data Finder record
(https://doi.org/10.25912/rdf_1749535700270) lists "Creative Commons Non-Commercial
(Any)". All three are non-commercial and none has a NoDerivatives term.

Synthetic radar dataset and the corpus's only radar source.

| | |
|---|---|
| Raw format | one HDF5 file, `RadChar-Baseline.h5` |
| Structure | `iq`: (1,000,000, 512) complex128; `labels`: (1,000,000,) structured array |
| Sample rate | 3.2 MHz, a constant in the config † |
| SNR | −20 to +20 dB in 1 dB steps, 41 levels, stored as `snr_db` |
| Classes (5) | `coherent_pulse_train`, `barker_code`, `polyphase_barker_code`, `frank_code`, `linear_frequency_modulated` |
| Windows | 1,000,000 × 512 samples, 200,000 per class; each row is already a window |
| Noise filtering | none; SNR is part of the design |
| Aggregation | start-cropped from 512 to 500 samples (12 dropped) |
| Group key | none; simulated i.i.d. |

**Raw label fields.** The structured `labels` array has seven fields; six are used.

| Raw field | Type | Stored as |
|---|---|---|
| `index` | int64 | not stored |
| `signal_type` | int64, 0 to 4 | `radar_signal_type`, through `SIGNAL_TYPES` |
| `number_of_pulses` | int64 | `number_of_pulses` |
| `pulse_width` | float64, seconds | `pulse_width_s` |
| `time_delay` | float64, seconds | `time_delay_s` |
| `pulse_repetition_interval` | float64, seconds | `pri_s` |
| `signal_to_noise_ratio` | int64, dB | `snr_db` |

**Radar labels.** The five classes are neither a technology nor a modulation, so they
are stored in `radar_signal_type`, and `waveform_type` holds the intra-pulse family.
The mapping and its basis are in `DESIGN.md` §1.2. In the aggregate, radar windows have
`technology_type = "unknown"` and `modulation_type = "unknown"`; they make up about 63%
of the corpus's `"unknown"` modulation windows, as expected.

**Metadata fields (10).** `radar_signal_type`, `waveform_type`, `snr_db`,
`number_of_pulses`, `pulse_width_s`, `time_delay_s`, `pri_s`, `sample_index`,
`sample_rate_hz`, `file_path`. `sample_index` is the row number in the raw file. The
per-dataset file has no `technology_type` or `modulation_type`. The constant sample
rate is written by declaring 3,200,000 as the fill of `sample_rate_hz`.

**Splitting.** No group key: examples are simulated independently, there is no
`recording_id` field in this file, and the aggregate fills it with `""`. A random split
is safe.

---

## Classes shared across sources

Every window in the aggregate carries `source_dataset`, so per-class accuracy can be
split by source:

| Class | Sources |
|---|---|
| WiFi | `sub_ghz`, `lte_wifi_dvbt`, `nist_wifi_bluetooth`, `its_band` |
| LTE | `lte_wifi_dvbt`, `its_band` |
| Noise | `sub_ghz`, `its_band` |
| 64QAM | `rml2016_10a`, `rml2018_01a`, `sub_ghz`, `nist_wifi_bluetooth` |
| BPSK, 16QAM | `rml2016_10a`, `rml2018_01a`, `sub_ghz` |
| QPSK, 8PSK | `rml2016_10a`, `rml2018_01a` |
| OQPSK | `rml2018_01a`, `sub_ghz` |
| 256QAM | `rml2018_01a`, `nist_wifi_bluetooth` |

The WiFi class is internally diverse: 802.11ah in `sub_ghz`, 802.11n and 802.11ac
elsewhere. `wifi_standard` in each per-dataset file tells them apart.

---

## Raw sample count

The paper's "source IQ samples" figure, 2.06 × 10¹⁰, is the number of complex samples
in the raw inputs as published, before windowing, noise filtering, padding or
cropping. It is distinct from the retained count, 25,836,227 windows × 500 =
1.29 × 10¹⁰ samples.

| Source | Raw IQ samples | Derivation |
|---|---|---|
| `nist_wifi_bluetooth` | 16,200,000,000 | 3 files × 180 captures × 30,000,000 samples |
| `rml2018_01a` | 2,617,245,696 | 2,555,904 rows × 1024 |
| `sub_ghz` | 1,024,000,000 | sum of `IQ_samples` lengths over the 125 raw `.mat` files |
| `radchar` | 512,000,000 | 1,000,000 rows × 512 |
| `lte_wifi_dvbt` | 207,671,522 | sum of the 190 `.bin` file sizes ÷ 8 bytes |
| `its_band` | 49,500,000 | 6 files × 7,500 observations × 1100 |
| `rml2016_10a` | 28,160,000 | 220,000 entries × 128 |
| **Total** | **20,638,577,218** | 2.06 × 10¹⁰ |

**The `sub_ghz` figure cannot be recovered from the built file.** That file records
only the 927,152 windows that survived noise filtering, out of 2,048,000 cut from the
raw data. The count comes from the raw `.mat` files, and this table is its record. It
can be recomputed without loading the data:

```python
import scipy.io, pathlib
sum(int(shape[0] * shape[1])
    for path in pathlib.Path("data/sub_ghz/raw").rglob("*.mat")
    for name, shape, _ in scipy.io.whosmat(str(path)) if name == "IQ_samples")
```

The `lte_wifi_dvbt` figure differs from 415,344 × 500 = 207,672,000 by 478 samples:
the zero-padding of each file's last window.
