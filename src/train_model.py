from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def train_model(data):
    feature_cols = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "remote_ratio",
        "company_location",
        "company_size"
    ]

    X = data[feature_cols]
    y = data["salary_in_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test
