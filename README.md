# WAGE//LAB — FIFA Player Wage Predictor

A redesigned Streamlit application for estimating FIFA player weekly wages using the original trained Random Forest model.

## What's included

- `app.py` — new WAGE//LAB application and interface
- `data/mean_wages.csv` — nationality wage benchmarks
- `data/nationality_mapping.csv` — exact nationality codes recovered from the original encoded training data
- `data/players.csv` — original encoded player reference data for market visualisations
- `requirements.txt` — pinned deployment dependencies
- `runtime.txt` — Python 3.11 runtime

## Important: model file

The original model (`wage_predictor_model`, approximately 110 MB) is intentionally not in this source ZIP. GitHub blocks regular Git files larger than 100 MiB. Upload it as an asset in a GitHub Release instead.

1. Create a public GitHub repository and upload the contents of this folder.
2. Open **Releases → Create a new release**.
3. Create a tag such as `v1.0.0`.
4. Attach the original `wage_predictor_model` file as a release asset.
5. Publish the release.
6. Copy the asset's direct download URL. It should look like:
   `https://github.com/USERNAME/REPOSITORY/releases/download/v1.0.0/wage_predictor_model`
7. In Streamlit Community Cloud, deploy `app.py`.
8. Open **App settings → Secrets** and add:

```toml
MODEL_URL = "https://github.com/USERNAME/REPOSITORY/releases/download/v1.0.0/wage_predictor_model"
```

The model is downloaded and cached when the app starts. For a private repository, a public release asset or an authenticated download mechanism is needed; do not put a GitHub token in source code.

## Run locally

Use Python 3.11:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Place `wage_predictor_model` in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

Alternatively, set `MODEL_URL` as an environment variable.

## Model compatibility

The model was originally serialized with scikit-learn 1.2.2. This project pins scikit-learn 1.2.2 and NumPy 1.26.4 and targets Python 3.11 to avoid loading it under a newer, incompatible scikit-learn release.

Only load pickle files you trust. Pickle deserialization can execute code.

## Feature order and encoding

The app passes inputs in this exact order:

1. `Overall`
2. `Potential`
3. `Age`
4. `Nationality`
5. `Attacking_WorkRate`
6. `Defensive_WorkRate`

Nationality codes and work-rate codes are preserved from the original encoded data. Work rates use Low=0, Medium=1, High=2.

## Notes

- The displayed wage is treated as EUR per week, consistent with the source FIFA wage field.
- This is an indicative model estimate, not an official player contract valuation.
- The model does not include club, league, contract length, bonuses or other commercial variables.
