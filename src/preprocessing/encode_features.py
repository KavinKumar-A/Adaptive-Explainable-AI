from sklearn.preprocessing import LabelEncoder

def encode(data):
    """
    Encode categorical features using LabelEncoder.
    """
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])
    return data