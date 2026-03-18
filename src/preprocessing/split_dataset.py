from sklearn.model_selection import train_test_split, StratifiedKFold


def split(data, target_col='Label'):
    """
    Split the dataset into train and test sets.
    Uses stratified splitting to maintain class distribution (Paper §7.3).
    """
    if target_col not in data.columns:
        possible_targets = ['label', 'Label', 'attack_type', 'class']
        for col in possible_targets:
            if col in data.columns:
                target_col = col
                break
        else:
            raise ValueError("Target column not found")

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Stratified 80/20 split preserving class proportions (Paper §7.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_stratified_kfold(n_splits=5):
    """
    Return a StratifiedKFold cross-validator for full evaluation (Paper §7.3):
    'Stratified K-Fold Cross-Validation (k=5) guarantees that each
     80% training / 20% testing split maintains the exact proportional
     representation of classes.'
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)