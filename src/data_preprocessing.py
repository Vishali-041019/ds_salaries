import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    return data

def clean_data(data):
    data = data.drop_duplicates()
    data = data.dropna()
    return data