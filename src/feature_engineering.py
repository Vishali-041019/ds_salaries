from sklearn.preprocessing import LabelEncoder

def encode_features(data):
    encoders = {}

    categorical_cols = [
        'experience_level',
        'employment_type',
        'job_title',
        'salary_currency',
        'employee_residence',
        'company_location',
        'company_size'
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    return data, encoders