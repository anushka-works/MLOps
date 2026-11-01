import pandas as pd
import numpy as np
from src.ml_utils import clean_and_impute, train_dummy_model

def test_clean_and_impute():
    df = pd.DataFrame({"feature1": [10, np.nan, 10]})
    result = clean_and_impute(df)
    assert result.isnull().sum().sum() == 0

def test_model_prediction():
    model = train_dummy_model()
    probs = model.predict_proba([[1], [2]])
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1)