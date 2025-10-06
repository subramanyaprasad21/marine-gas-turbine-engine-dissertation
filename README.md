# XGBoost Prediction API (Lightweight Models)

This package contains lightweight XGBoost models (trained on a 20% sample for speed) and a simple Flask API to serve predictions.

## Files
- `app.py` - Flask API that loads models and exposes `/predict` POST endpoint.
- `xgb_*` - Pickled XGBoost model files for each target.
- `predict_example.py` - Example client that calls the API.
- `requirements.txt` - Python dependencies.

## Usage
1. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Put your full `data.csv` in the same folder if you plan to retrain.
4. Run the app: `python app.py`
5. Send POST requests to `http://localhost:5000/predict` with JSON payload containing the feature columns.

Note: The models here are lightweight (sample-trained) for quick deployment/testing. For production, retrain on full dataset using the training code provided below.

## Retrain on full dataset (recommended for production)
```python
from xgboost import XGBRegressor
import joblib
import pandas as pd

df = pd.read_csv('data.csv')
# apply same column cleaning as used here, define targets and features, then:
model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X, y)
joblib.dump(model, 'xgb_fullmodel.pkl')
```
