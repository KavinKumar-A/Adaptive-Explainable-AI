import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import warnings


def remove_correlated_features(X, threshold=0.90):
    """
    Remove highly correlated features (Pearson > threshold).
    Per Paper §7.2: 'Highly correlated features (Pearson's correlation
    coefficient > 0.90) are pruned.'
    """
    if hasattr(X, 'values'):
        corr_matrix = np.abs(np.corrcoef(X.values, rowvar=False))
    else:
        corr_matrix = np.abs(np.corrcoef(X, rowvar=False))

    n_features = corr_matrix.shape[0]
    cols_to_drop = set()

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if corr_matrix[i, j] > threshold and j not in cols_to_drop:
                cols_to_drop.add(j)

    cols_to_keep = [i for i in range(n_features) if i not in cols_to_drop]

    if hasattr(X, 'iloc'):
        return X.iloc[:, cols_to_keep], cols_to_keep
    else:
        return X[:, cols_to_keep], cols_to_keep


def select_features(X, y, k=20):
    """
    Two-stage feature selection per Paper §7.2:
    1. Remove highly correlated features (Pearson > 0.90).
    2. Select top-k features using ANOVA F-test (Information Gain proxy).
    """
    # Stage 1: Correlation pruning
    X_pruned, kept_indices = remove_correlated_features(X, threshold=0.90)

    # Stage 2: ANOVA F-test selection on remaining features
    actual_k = min(k, X_pruned.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=f_classif, k=actual_k)
        X_selected = selector.fit_transform(X_pruned, y)

    return X_selected, selector, kept_indices