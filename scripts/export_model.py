"""Export the saved scikit-learn forest for browser-side inference.

Each node is 12 little-endian bytes: feature (uint16), right child (uint16),
and threshold or leaf value (float64). A leaf uses feature=65535. The left
child of an internal node is always the next node in sklearn's tree order.
"""

from __future__ import annotations

import csv
import json
import struct
from collections import defaultdict
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "site" / "dist"
MODEL_PATH = ROOT / "bandgap_random_forest.joblib"
CSV_PATH = ROOT / "expt_gap.csv"


def main() -> None:
    DIST.mkdir(exist_ok=True)
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    elements = list(saved["feature_columns"])
    if len(elements) != model.n_features_in_:
        raise ValueError("Saved feature list does not match model input count")

    offsets = [0]
    node_count = 0
    with (DIST / "forest.bin").open("wb") as output:
        for estimator in model.estimators_:
            tree = estimator.tree_
            if tree.node_count > 65535:
                raise ValueError("A tree has too many nodes for uint16 child indices")
            for index in range(tree.node_count):
                feature = int(tree.feature[index])
                if feature < 0:
                    output.write(struct.pack("<HHd", 65535, 0, float(tree.value[index, 0, 0])))
                else:
                    left = int(tree.children_left[index])
                    right = int(tree.children_right[index])
                    if left != index + 1:
                        raise ValueError("Tree nodes are not in preorder")
                    output.write(struct.pack("<HHd", feature, right, float(tree.threshold[index])))
            node_count += tree.node_count
            offsets.append(node_count)

    metadata = {
        "format": "rf-preorder-v1",
        "node_bytes": 12,
        "elements": elements,
        "tree_offsets": offsets,
        "tree_count": len(model.estimators_),
        "node_count": node_count,
        "unit": "eV",
    }
    (DIST / "model-meta.json").write_text(json.dumps(metadata, separators=(",", ":")))

    readings: dict[str, list[float]] = defaultdict(list)
    with CSV_PATH.open(newline="") as source:
        for row in csv.DictReader(source):
            try:
                readings[row["formula"].strip()].append(float(row["gap expt"]))
            except (ValueError, KeyError):
                continue
    observations = {
        formula: {"mean": sum(values) / len(values), "count": len(values)}
        for formula, values in readings.items()
    }
    (DIST / "observations.json").write_text(json.dumps(observations, separators=(",", ":")))
    print(f"Exported {len(model.estimators_)} trees and {node_count:,} nodes")
    print(f"Exported {sum(map(len, readings.values())):,} experimental entries")


if __name__ == "__main__":
    main()
