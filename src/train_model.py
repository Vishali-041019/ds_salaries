from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def train_model(data):

    X = data.drop(['salary', 'salary_currency', 'salary_in_usd'], axis=1)
    y = data['salary_in_usd']

    # Print feature names
    print("Training Features:")
    print(X.columns.tolist())
    print("Number of Features:", len(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Create Scaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test