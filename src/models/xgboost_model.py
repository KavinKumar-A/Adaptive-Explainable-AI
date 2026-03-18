from xgboost import XGBClassifier

def get_model():
    """
    Return an XGBoost model.
    """
    return XGBClassifier(random_state=42)