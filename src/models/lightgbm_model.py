from lightgbm import LGBMClassifier


def get_model():
    """
    Return a LightGBM model (Paper §6.2.3).
    Uses leaf-wise tree growth for optimal speed/memory on large datasets.
    """
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=-1,        # No limit (leaf-wise growth)
        num_leaves=31,
        random_state=42,
        verbose=-1           # Suppress verbose output
    )
