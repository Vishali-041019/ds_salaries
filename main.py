from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_features
from src.visualization import (
    salary_histogram,
    salary_boxplot,
    experience_countplot,
    company_size_barplot,
    remote_salary_scatter,
    salary_violinplot,
    salary_year_lineplot,
    company_size_piechart,
    correlation_heatmap,
    salary_kdeplot
)
from src.train_model import train_model
from src.evaluate_model import evaluate_model

import pickle
import os

# Load Data
data = load_data("data/ds_salaries.csv")

# Data Cleaning
data = clean_data(data)

# Feature Engineering
data, encoders = encode_features(data)

# Data Visualization
salary_histogram(data)
salary_boxplot(data)
experience_countplot(data)
company_size_barplot(data)
remote_salary_scatter(data)
salary_violinplot(data)
salary_year_lineplot(data)
company_size_piechart(data)
correlation_heatmap(data)
salary_kdeplot(data)

# Model Training
model, scaler, X_test, y_test = train_model(data)

# Model Evaluation
evaluate_model(model, X_test, y_test)

# Create models folder if it doesn't exist
os.makedirs("artifacts/models", exist_ok=True)

with open("artifacts/models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("artifacts/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully.")