"""
Persistent identifiers of the RadioPile release.

Zenodo mints a DOI when a deposition is created, before it is published ("Reserve
DOI" in the deposit form). Reserve first, record here, then build and upload,
because these values land in the HDF5 root attributes: recording them after
publishing would change the file the DOI identifies.

scripts/_set_release_identifiers.py writes the values below. The code DOI is a
manually uploaded software record, not the GitHub integration, so it can be
reserved before the build as well.
"""

RECORD_A_DOI = "10.5281/zenodo.22876704"   # the aggregated corpus
RECORD_B_DOI = "10.5281/zenodo.22877108"   # the per-dataset files, published separately
CODE_DOI     = "10.5281/zenodo.23174413"   # this pipeline, the tagged source archive

# Version of the release, in semantic-versioning form: the git tag (v1.0.0),
# `version` in CITATION.cff and in the Croissant sidecars. The collection is
# referred to as "RadioPile v1.0" for short.
RELEASE_VERSION = "1.0.0"

# Publication date of the two data records, written as `datePublished` in the
# Croissant sidecars. Same date as `date-released` in CITATION.cff.
RELEASE_DATE = "2026-10-12"
