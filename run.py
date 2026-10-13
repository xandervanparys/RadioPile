#!/usr/bin/env python
"""Build a dataset by name. Uses the registry to resolve build modules."""

import argparse
import importlib

from datasets.registry import list_datasets, get_dataset


# Non-registry build targets (aggregation has its own build module)
EXTRA_TARGETS = {
    "aggregation": "datasets.aggregation.build_dataset",
}


def main():
    all_targets = sorted(set(list_datasets()) | set(EXTRA_TARGETS.keys()))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=all_targets, help="dataset to build")
    target = parser.parse_args().dataset

    if target in EXTRA_TARGETS:
        module = importlib.import_module(EXTRA_TARGETS[target])
    else:
        entry = get_dataset(target)
        module = importlib.import_module(f"{entry.build_module}.build_dataset")

    module.main()


if __name__ == "__main__":
    main()
