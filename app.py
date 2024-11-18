from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load('ridge_model.pkl')
poly = joblib.load('polynomial_transformer.pkl')

def predict_corrected_rank(percentile: float, total_candidates: int) -> float:
    # Calculate initial predicted rank using the formula
    predicted_rank = ((100 - percentile) * total_candidates) / 100
    # Predict correction factor using the polynomial regression model
    percentile_poly = poly.transform([[percentile]])
    predicted_correction = model.predict(percentile_poly)[0][0]
    # Adjust the predicted rank with the correction factor
    corrected_rank = predicted_rank + predicted_correction
    # Ensure the rank does not exceed the total number of candidates or become negative
    corrected_rank = max(1, min(corrected_rank, total_candidates))
    return corrected_rank


@app.get("/predict")
def get_corrected_rank(percentile: float, total_candidates: int):
    corrected_rank = predict_corrected_rank(percentile, total_candidates)
    return {"percentile": percentile, "total_candidates": total_candidates, "corrected_rank": corrected_rank}


