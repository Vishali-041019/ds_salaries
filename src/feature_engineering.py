import pandas as pd

def encode_features(data):
    """
    Encode categorical columns using fixed manual mappings
    so app.py dropdown values match training exactly.
    """

    # ---------------- FIXED ENCODERS FOR APP DROPDOWNS ----------------
    encoders = {
        "experience_level": {
            "Entry": 0,
            "Mid": 1,
            "Senior": 2,
            "Executive": 3
        },
        "employment_type": {
            "Full-time": 0,
            "Part-time": 1,
            "Contract": 2,
            "Freelance": 3
        },
        "company_size": {
            "Small": 0,
            "Medium": 1,
            "Large": 2
        },
        "job_title": {
            "Data Analyst": 0,
            "Data Scientist": 1,
            "Machine Learning Engineer": 2,
            "Data Engineer": 3,
            "BI Analyst": 4,
            "Research Scientist": 5
        },
        "employee_residence": {
            "United States": 0,
            "India": 1,
            "United Kingdom": 2,
            "Germany": 3,
            "Canada": 4,
            "France": 5
        },
        "company_location": {
            "United States": 0,
            "India": 1,
            "United Kingdom": 2,
            "Germany": 3,
            "Canada": 4,
            "France": 5
        }
    }

    # ---------------- CLEAN DATA VALUES TO MATCH APP ----------------
    # ds_salaries dataset often contains short codes like EN, FT, US, etc.
    # Convert them into the full names used in app.py dropdowns.

    experience_clean_map = {
        "EN": "Entry",
        "MI": "Mid",
        "SE": "Senior",
        "EX": "Executive",
        "Entry": "Entry",
        "Mid": "Mid",
        "Senior": "Senior",
        "Executive": "Executive"
    }

    employment_clean_map = {
        "FT": "Full-time",
        "PT": "Part-time",
        "CT": "Contract",
        "FL": "Freelance",
        "Full-time": "Full-time",
        "Part-time": "Part-time",
        "Contract": "Contract",
        "Freelance": "Freelance"
    }

    company_size_clean_map = {
        "S": "Small",
        "M": "Medium",
        "L": "Large",
        "Small": "Small",
        "Medium": "Medium",
        "Large": "Large"
    }

    residence_clean_map = {
        "US": "United States",
        "IN": "India",
        "GB": "United Kingdom",
        "DE": "Germany",
        "CA": "Canada",
        "FR": "France",
        "United States": "United States",
        "India": "India",
        "United Kingdom": "United Kingdom",
        "Germany": "Germany",
        "Canada": "Canada",
        "France": "France"
    }

    location_clean_map = {
        "US": "United States",
        "IN": "India",
        "GB": "United Kingdom",
        "DE": "Germany",
        "CA": "Canada",
        "FR": "France",
        "United States": "United States",
        "India": "India",
        "United Kingdom": "United Kingdom",
        "Germany": "Germany",
        "Canada": "Canada",
        "France": "France"
    }

    # ---------------- APPLY CLEANING ----------------
    if "experience_level" in data.columns:
        data["experience_level"] = data["experience_level"].replace(experience_clean_map)

    if "employment_type" in data.columns:
        data["employment_type"] = data["employment_type"].replace(employment_clean_map)

    if "company_size" in data.columns:
        data["company_size"] = data["company_size"].replace(company_size_clean_map)

    if "employee_residence" in data.columns:
        data["employee_residence"] = data["employee_residence"].replace(residence_clean_map)

    if "company_location" in data.columns:
        data["company_location"] = data["company_location"].replace(location_clean_map)

    # ---------------- KEEP ONLY APP-SUPPORTED VALUES ----------------
    # Because your app dropdown only contains these values

    allowed_job_titles = list(encoders["job_title"].keys())
    allowed_residences = list(encoders["employee_residence"].keys())
    allowed_locations = list(encoders["company_location"].keys())

    if "job_title" in data.columns:
        data = data[data["job_title"].isin(allowed_job_titles)].copy()

    if "employee_residence" in data.columns:
        data = data[data["employee_residence"].isin(allowed_residences)].copy()

    if "company_location" in data.columns:
        data = data[data["company_location"].isin(allowed_locations)].copy()

    # ---------------- APPLY FIXED ENCODING ----------------
    data["experience_level"] = data["experience_level"].map(encoders["experience_level"])
    data["employment_type"] = data["employment_type"].map(encoders["employment_type"])
    data["company_size"] = data["company_size"].map(encoders["company_size"])
    data["job_title"] = data["job_title"].map(encoders["job_title"])
    data["employee_residence"] = data["employee_residence"].map(encoders["employee_residence"])
    data["company_location"] = data["company_location"].map(encoders["company_location"])

    # ---------------- DROP UNUSED COLUMN ----------------
    # salary_currency is not used in your app input
    if "salary_currency" in data.columns:
        data = data.drop(columns=["salary_currency"])

    # ---------------- DROP ROWS WITH MISSING ENCODING ----------------
    data = data.dropna().copy()

    # ---------------- CONVERT TO INTEGER ----------------
    encoded_cols = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "company_location",
        "company_size"
    ]

    for col in encoded_cols:
        data[col] = data[col].astype(int)

    return data, encoders
