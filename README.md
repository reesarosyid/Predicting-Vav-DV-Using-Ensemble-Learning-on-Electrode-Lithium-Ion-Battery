# Machine Learning Prediction of Battery Electrode Properties

Data pipeline and machine learning models that predict two properties of metal-ion battery electrodes from composition and structure features:

- **Average voltage** (`V_av`, V)
- **Volume change** (`dV_percent`, %)

Materials come from the [Materials Project](https://materialsproject.org) and [AFLOW](http://aflow.org). The pipeline pairs charged and discharged structures, computes their electrochemical properties and featurizes them with Magpie descriptors (matminer).

Supported working ions: Li, Na, K, Rb, Cs, Mg, Ca, Al, Zn, Y (see `config.py`).

## Repository structure

| File | Description |
|------|-------------|
| `config.py` | Working ions, valences, constants and dataset paths |
| `scraping.py` | Data pipeline: fetch → pair → properties → features |
| `notebook_vav.ipynb` | Modelling of average voltage (`V_av`) |
| `notebook_dv.ipynb` | Modelling of volume change (`dV_percent`) |
| `requirements.txt` | Pinned Python dependencies |

## Installation

Requires Python 3.12. The notebooks use RAPIDS cuML (GPU random forest), which needs an NVIDIA GPU with CUDA 12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data pipeline

Step 1 downloads data and needs a Materials Project API key ([get one here](https://next-gen.materialsproject.org/api)):

```bash
export MP_API_KEY="your_api_key"
python scraping.py                 # run all steps
```

| Step | What it does | Output |
|------|--------------|--------|
| 1 | Fetch materials from MP and AFLOW, element reference energies | `Dataset/01`–`04_*.json` |
| 2 | Find charged/discharged pairs | `Dataset/05_pairs.json` |
| 3 | Compute properties | `Dataset/06_properties.csv` |
| 4 | Compute Magpie features | `Dataset/07_features.csv` |

Steps 2–4 run offline. Useful options:

```bash
python scraping.py --steps 2,3,4                                   # skip downloading
python scraping.py --steps 4 --features-out Dataset/07_features.parquet
python scraping.py --steps 4 --max-pairs 5000                      # quick test run
python scraping.py --validate                                      # energy consistency check only
python scraping.py --help
```

### Computed properties

For a pair with working ion of valence *q* inserted from *x₁* (charged) to *x₂* (discharged):

- Average voltage: `V_av = -(Ef_disch - Ef_chg) / (q · (x₂ - x₁))`
- Volume change: `ΔV% = |V_disch - V_chg| / min(V_disch, V_chg) × 100`
- Specific capacity (mAh/g): `q · (x₂ - x₁) · F / (3.6 · M)`
- Specific energy (Wh/kg): `capacity × V_av`

## Modelling

Each notebook covers EDA, preprocessing, model training (Random Forest, XGBoost, LightGBM, CatBoost) with 10-fold cross-validation and grid search, test-set evaluation, SHAP analysis, partial dependence plots and an applicability domain analysis (k-nearest neighbours).

The notebooks read `../Dataset/Dataset_Li.csv`. Set `DATA_PATH` in the *Config* cell to your own dataset location.

## Data availability

The datasets are not in this repository because of their size. Generate them with `scraping.py` as described above.

## Citation

If you use this code, please cite:

```
[Citation will be added after publication]
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

