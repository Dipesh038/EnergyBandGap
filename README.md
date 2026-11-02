# Band Gap Lab

An interactive website for presenting the saved `bandgap_random_forest.joblib` model. Enter a chemical formula to see its predicted energy band gap, atomic fractions, and an illustrative band diagram. If the exact formula appears in `expt_gap.csv`, the page also shows the CSV reference and its absolute difference from the prediction.

## Run locally

The `site/dist/` folder is a static website. Serve it over HTTP, then open the shown URL:

```bash
python3 -m http.server 8000 --directory site/dist
```

The first prediction downloads the exported forest (about 7 MB). Predictions then run in the browser, so no Python server or internet connection is required after the files are served. The font falls back to system fonts when offline.

## Rebuild the browser model

The checked-in browser model is generated from the supplied joblib file. To regenerate it after replacing the saved model or CSV:

```bash
python3 -m venv .venv
.venv/bin/pip install "joblib>=1.4,<2" "numpy>=2,<3" "scipy>=1.15,<2" "scikit-learn>=1.6,<2"
.venv/bin/python scripts/export_model.py
```

The original model was saved with scikit-learn 1.6.1. Load joblib files only from trusted sources. The site uses each element's atom count divided by the total atom count as its input vector, in the `feature_columns` order stored in the joblib file. It averages all 200 trees exactly as the random forest does.

Supported inputs include `ZnO`, `GaAs`, `Si`, `Hg0.7Cd0.3Te`, and groups such as `Ca(OH)2`. Unicode subscripts are accepted, and uppercase formulas are corrected when the element split is unambiguous (`HCL` becomes `HCl`). Use standard element capitalization for ambiguous formulas (`CO` means carbon and oxygen, while `Co` means cobalt). Hydrates, charges, and leading formula coefficients are not supported.

The CSV reference is descriptive, not a held-out test set. No accuracy metric is claimed.
