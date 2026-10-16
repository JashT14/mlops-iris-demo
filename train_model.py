from pycaret.classification import setup, compare_models
from pycaret.datasets import get_data
import joblib

iris = get_data("iris")

setup(
    data=iris,
    target="species",
    session_id=42,
    verbose=False,
    html=False
)

model = compare_models()
joblib.dump(model, "iris_model.pkl")

print("Model trained and saved: iris_model.pkl")
