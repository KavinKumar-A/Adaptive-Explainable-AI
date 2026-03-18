from sklearn.ensemble import RandomForestClassifier

def get_model():
    """
    Return a Random Forest model.
    """
    return RandomForestClassifier(n_estimators=100, random_state=42)