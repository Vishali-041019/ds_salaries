# Data Science Salary Prediction

## Project Overview

This project predicts salaries in the Data Science field using Machine Learning techniques. The model is trained on a dataset containing information about experience level, employment type, job title, company size, remote work ratio, and other factors that influence salary.

The project includes:

* Data Preprocessing
* Feature Engineering
* Data Visualization
* Model Training
* Model Evaluation
* Model Export using Pickle

---

## Dataset

The dataset contains information about Data Science professionals and their salaries.

### Features

* work_year
* experience_level
* employment_type
* job_title
* salary
* salary_currency
* salary_in_usd
* employee_residence
* remote_ratio
* company_location
* company_size

### Target Variable

* salary_in_usd

---

## Project Structure

```text
data-science-salary-prediction/
│
├── data/
│   └── ds_salaries.csv
│
├── artifacts/
│   └── model.pkl
│
├── plots/
│   └── generated visualization images
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── visualization.py
│   └── plots.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Pickle

---

## Exploratory Data Analysis

The project includes:

* Salary Distribution Histogram
* Salary Box Plot
* Experience Level Count Plot
* Company Size Bar Plot
* Remote Ratio vs Salary Scatter Plot
* Salary Violin Plot
* Salary Trend by Year
* Company Size Pie Chart
* Correlation Heatmap
* Salary Density Curve

Generated plots are saved inside the `plots/` folder.

---

## Model Training Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Data Visualization
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Save Trained Model

---

## Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Model Export

The trained model is saved as:

```text
artifacts/model.pkl
```

This model can be loaded later for salary prediction without retraining.

---

## Future Improvements

* Hyperparameter Tuning
* Feature Selection Optimization
* Streamlit Web Application
* Multiple Model Comparison
* Cloud Deployment

---

## Author

Vishali

Machine Learning & Data Science Enthusiast
