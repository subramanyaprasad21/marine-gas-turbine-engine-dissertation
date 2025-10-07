# RandomForest Prediction API

This package contains RandomForest models (trained on the full dataset) and a Flask API.

## Files
- `app.py` - Flask API that loads models and exposes `/predict` POST endpoint.
- `rf_*` - Pickled RandomForest model files for each target.
- `requirements.txt` - Python dependencies (no xgboost).

