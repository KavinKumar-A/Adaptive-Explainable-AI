from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import time


def explain(model, X_train, X_test, instance_idx=0, ds_name="dataset", model_name="model"):
    """
    Generate LIME explanation for a specific instance with latency
    benchmarking per Paper §7.4.
    """
    # Handle both DataFrame and numpy array inputs
    if hasattr(X_train, 'values'):
        train_data = X_train.values
        feature_names = list(X_train.columns)
    else:
        train_data = np.array(X_train)
        feature_names = [f"feature_{i}" for i in range(train_data.shape[1])]

    if hasattr(X_test, 'iloc'):
        test_instance = X_test.iloc[instance_idx].values
    elif hasattr(X_test, '__getitem__'):
        test_instance = np.array(X_test[instance_idx])
    else:
        test_instance = X_test[instance_idx]

    explainer = LimeTabularExplainer(
        train_data,
        feature_names=feature_names,
        class_names=['normal', 'attack'],
        discretize_continuous=True
    )

    # --- Latency Benchmarking (Paper §7.4) ---
    start_time = time.time()
    exp = explainer.explain_instance(
        test_instance,
        model.predict_proba,
        num_features=10
    )
    elapsed = time.time() - start_time

    # Save explanation as HTML
    os.makedirs('reports/figures', exist_ok=True)
    exp.save_to_file(f'reports/figures/{ds_name}_{model_name}_lime_instance_{instance_idx}.html')

    results = {
        "explanation": exp,
        "feature_weights": exp.as_list(),
        "latency_ms": round(elapsed * 1000, 3),
    }
    return results