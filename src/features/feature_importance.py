from sklearn.ensemble import RandomForestClassifier

def get_importance(X, y):
    """
    Get feature importance using Random Forest.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importance = model.feature_importances_
    return importance