import pandas as pd
from sklearn.dummy import DummyClassifier

def clean_and_impute(df):
    return df.fillna(df.mean())

def train_dummy_model():
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[1], [2], [3]], [0, 1, 1])
    return model