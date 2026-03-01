import numpy as np


def load_features(filepath):
    data = np.genfromtxt(filepath, delimiter=',', dtype=object, skip_header=1)
    headers = np.genfromtxt(filepath, delimiter=',', dtype=str, max_rows=1)

    tenure_index = list(headers).index("tenure")
    monthly_index = list(headers).index("MonthlyCharges")
    churn_index = list(headers).index("Churn")
    contract_index = list(headers).index("Contract")

    tenure = data[:, tenure_index].astype(float)
    monthly = data[:, monthly_index].astype(float)
    churn = ( np.char.strip(data[:, churn_index].astype(str)) == "Yes").astype(int)
    contract_raw = np.char.strip(data[:,contract_index].astype(str))
    contract_month = (contract_raw == "Month-to-month").astype(int)
    contract_year = (contract_raw == "One year").astype(int)



    X = np.column_stack((tenure, monthly,contract_month,contract_year))
    y = churn

    return X, y