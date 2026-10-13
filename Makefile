# RadioPile release Makefile
#
# Targets:
#   make release         build a versioned release bundle in $(RELEASE_DIR)
#   make validate        run the dataset validator (no report file)
#   make clean-release   remove $(RELEASE_DIR)
#
# What `release` does:
#   * COPIES the HDF5s into the bundle, never hardlinks them. h5py truncates an
#     existing file in place on rebuild, so a hardlinked bundle would be
#     overwritten by the next `python run.py <dataset>`.
#   * Names the aggregated HDF5 `radiopile_v1.0.h5` (AGG_BUNDLE_NAME below).
#   * Copies per-dataset artifacts under their REGISTRY name, the value the
#     corpus stores in `source_dataset` and the argument `run.py` takes.
#   * Requires the `.h5.config.json` snapshot each build wrote; bundling stops
#     if one is missing, because a snapshot made up afterward would not record
#     the commit the file was built from.
#   * Copies each dataset's `config.py` in beside its artifacts, so every
#     artifact directory is self-contained.
#   * Rewrites each bundled `*.croissant.json` to point at its sibling HDF5
#     (scripts/_fix_bundle_croissant.py).
#   * REFUSES to write into an existing non-empty $(RELEASE_DIR). Pass FORCE=1
#     to replace one deliberately.
#

RELEASE_VERSION := v1.0
RELEASE_DIR     := release/$(RELEASE_VERSION)
AGG_BUNDLE_NAME := radiopile_$(RELEASE_VERSION)

PYTHON := $(shell test -x .venv_pipeline/bin/python && echo .venv_pipeline/bin/python || echo python3)

AGG_SRC_DIR := data/aggregation/processed

.PHONY: release validate clean-release

validate:
	@echo "==> Validating all built datasets"
	$(PYTHON) scripts/validate_datasets.py

release: validate _bundle
	@echo "==> Bundle ready at $(RELEASE_DIR)"
	@du -sh $(RELEASE_DIR)

_bundle:
	@if [ -d "$(RELEASE_DIR)" ] && [ -n "`ls -A '$(RELEASE_DIR)' 2>/dev/null`" ] && [ "$(FORCE)" != "1" ]; then \
	    echo "ERROR: $(RELEASE_DIR) already exists and is not empty."; \
	    echo "       Refusing to overwrite a populated release directory."; \
	    echo "       Either pick a new RELEASE_DIR=..., or re-run with FORCE=1"; \
	    echo "       to replace it (this deletes the existing bundle)."; \
	    exit 1; \
	fi
	@echo "==> Building release bundle at $(RELEASE_DIR)"
	rm -rf $(RELEASE_DIR)
	mkdir -p $(RELEASE_DIR)/aggregated
	mkdir -p $(RELEASE_DIR)/per_dataset

	@echo "--> Copying aggregated artifacts (renaming to $(AGG_BUNDLE_NAME))"
	@if [ ! -f "$(AGG_SRC_DIR)/aggregated.h5" ]; then \
	    echo "ERROR: aggregated HDF5 not found at $(AGG_SRC_DIR)/aggregated.h5"; \
	    echo "       Build it first: python run.py aggregation"; \
	    exit 1; \
	fi
	cp "$(AGG_SRC_DIR)/aggregated.h5" \
	    "$(RELEASE_DIR)/aggregated/$(AGG_BUNDLE_NAME).h5"
	@if [ -f "$(AGG_SRC_DIR)/aggregated.croissant.json" ]; then \
	    cp "$(AGG_SRC_DIR)/aggregated.croissant.json" \
	        "$(RELEASE_DIR)/aggregated/$(AGG_BUNDLE_NAME).croissant.json"; \
	fi
	@if [ -f "$(AGG_SRC_DIR)/aggregated.h5.config.json" ]; then \
	    cp "$(AGG_SRC_DIR)/aggregated.h5.config.json" \
	        "$(RELEASE_DIR)/aggregated/$(AGG_BUNDLE_NAME).h5.config.json"; \
	else \
	    echo "ERROR: $(AGG_SRC_DIR)/aggregated.h5.config.json is missing."; \
	    echo "       Rebuild the aggregate so its config snapshot is written."; \
	    exit 1; \
	fi

	@echo "--> Copying per-dataset artifacts"
	@for dir in data/*/processed/; do \
	    name=`basename \`dirname "$$dir"\``; \
	    if [ "$$name" = "aggregation" ]; then continue; fi; \
	    target="$(RELEASE_DIR)/per_dataset/$$name"; \
	    h5_src=`ls "$$dir"*.h5 2>/dev/null | head -1`; \
	    if [ -z "$$h5_src" ]; then \
	        echo "    WARN: no .h5 in $$dir, skipping $$name"; \
	        continue; \
	    fi; \
	    mkdir -p "$$target"; \
	    src_name=`basename "$$h5_src"`; \
	    if [ "$$src_name" != "$$name.h5" ]; then \
	        echo "    $$name <- $$src_name  (renamed)"; \
	    else \
	        echo "    $$name <- $$src_name"; \
	    fi; \
	    cp "$$h5_src" "$$target/$$name.h5"; \
	    if [ -f "$$dir$$src_name.config.json" ]; then \
	        cp "$$dir$$src_name.config.json" "$$target/$$name.h5.config.json"; \
	    else \
	        echo "ERROR: $$dir$$src_name.config.json is missing."; \
	        echo "       Rebuild $$name so its config snapshot is written."; \
	        exit 1; \
	    fi; \
	    croissant=`ls "$$dir"*.croissant.json 2>/dev/null | head -1`; \
	    if [ -n "$$croissant" ]; then \
	        cp "$$croissant" "$$target/$$name.croissant.json"; \
	    fi; \
	done

	@echo "--> Copying configuration source files beside their artifacts"
	@for cfg in datasets/*/config.py; do \
	    name=`basename \`dirname "$$cfg"\``; \
	    if [ "$$name" = "_template" ]; then continue; fi; \
	    if [ "$$name" = "aggregation" ]; then \
	        cp "$$cfg" "$(RELEASE_DIR)/aggregated/aggregation_config.py"; \
	    elif [ -d "$(RELEASE_DIR)/per_dataset/$$name" ]; then \
	        cp "$$cfg" "$(RELEASE_DIR)/per_dataset/$$name/$${name}_config.py"; \
	    else \
	        echo "    WARN: no bundled artifacts for $$name, skipping its config"; \
	    fi; \
	done

	@echo "--> Rewriting Croissant distribution refs to bundle-relative filenames"
	$(PYTHON) scripts/_fix_bundle_croissant.py $(RELEASE_DIR)

	@echo "--> Writing validation report"
	$(PYTHON) scripts/validate_datasets.py --fingerprint \
	    --save-report $(RELEASE_DIR)/validation_report.json

	@echo "--> Copying README and LICENSES from release_templates/"
	cp release_templates/RELEASE_README.md $(RELEASE_DIR)/README.md
	cp release_templates/LICENSES.md $(RELEASE_DIR)/LICENSES.md

clean-release:
	rm -rf $(RELEASE_DIR)
	@echo "==> Removed $(RELEASE_DIR)"
