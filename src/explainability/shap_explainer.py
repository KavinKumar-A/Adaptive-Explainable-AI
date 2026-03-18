import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def explain(model, X_test, ds_name="dataset", model_name="model"):
    """
    Generate SHAP explanations, save plots, and compute fidelity + latency
    per Paper §7.4.
    """
    # Select appropriate SHAP explainer
    model_type = type(model).__name__
    if model_type in ('RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier'):
        explainer = shap.TreeExplainer(model)
    else:
        # Correction #1: Reduce KernelExplainer background to 10-20 samples
        # for faster SHAP runtime on DNN models
        bg_size = 15
        if hasattr(X_test, 'values'):
            background = X_test[:bg_size] if len(X_test) > bg_size else X_test
        else:
            background = X_test[:bg_size] if X_test.shape[0] > bg_size else X_test
        explainer = shap.KernelExplainer(model.predict_proba, background)

    # Correction #2: Use a representative subset (100-200 samples) for SHAP
    subset_size = min(150, X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test))
    X_subset = X_test[:subset_size] if hasattr(X_test, 'iloc') else X_test[:subset_size]

    # --- Latency Benchmarking (Paper §7.4) ---
    start_time = time.time()
    shap_values = explainer.shap_values(X_subset)
    elapsed = time.time() - start_time
    n_samples = subset_size
    latency_per_flow = (elapsed / n_samples) * 1000  # milliseconds

    # --- Save Summary Plot ---
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure()
    # Correction #4: Use predicted-class SHAP values, not hardcoded index [1]
    sv = _get_class_shap_values(model, X_subset, shap_values)
    shap.summary_plot(sv, X_subset, show=False)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{ds_name}_{model_name}_shap_summary.png', bbox_inches='tight', dpi=150)
    plt.close()

    # --- Explanation Fidelity (Paper §7.4) ---
    fidelity = compute_explanation_fidelity(model, X_subset, sv, top_k=3)

    results = {
        "shap_values": shap_values,
        "latency_total_sec": round(elapsed, 3),
        "latency_per_flow_ms": round(latency_per_flow, 3),
        "explanation_fidelity": round(fidelity, 4),
    }
    return results


def _get_class_shap_values(model, X, shap_values):
    """
    Correction #4: Safely extract SHAP values for the predicted class
    instead of hardcoding shap_values[1] (binary assumption).
    For multi-class, uses the SHAP values corresponding to each sample's
    predicted class. For single-output, returns as-is.
    """
    if not isinstance(shap_values, list):
        return shap_values

    n_classes = len(shap_values)
    if n_classes == 2:
        # Binary: use positive-class (index 1) SHAP values
        return shap_values[1]
    else:
        # Multi-class: use SHAP values for the predicted class per sample
        predictions = model.predict(X)
        n_samples = shap_values[0].shape[0]
        sv_combined = np.zeros_like(shap_values[0])
        for i in range(n_samples):
            pred_class = int(predictions[i])
            if pred_class < n_classes:
                sv_combined[i] = shap_values[pred_class][i]
            else:
                sv_combined[i] = shap_values[0][i]
        return sv_combined


def compute_explanation_fidelity(model, X_test, shap_values, top_k=3):
    """
    Explanation Fidelity metric (Paper §7.4):
    Perturb the Top-k features identified by SHAP as most important.
    If the model changes its prediction, fidelity is high.

    Corrections applied:
    - #3: Perturb with feature mean/median instead of zero
    - #5: Use randomized sampling instead of first-N ordering
    - #8: Threshold-based importance selection instead of fixed top-k
    """
    if hasattr(X_test, 'values'):
        X_arr = X_test.values.copy()
    else:
        X_arr = np.array(X_test).copy()

    sv = np.array(shap_values)
    if sv.ndim == 3:
        # sv has shape (n_samples, n_features, n_classes)
        # Fix: Convert to list of (n_samples, n_features) arrays so _get_class_shap_values works correctly
        n_classes = sv.shape[2]
        sv_list = [sv[:, :, i] for i in range(n_classes)]
        sv = _get_class_shap_values(model, X_test, sv_list)
        sv = np.array(sv)

    # Correction #3: Compute feature medians for realistic perturbation
    feature_medians = np.median(X_arr, axis=0)

    # Correction #5: Randomized sampling to avoid ordering bias
    n_total = X_arr.shape[0]
    n_eval = min(100, n_total)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_total, size=n_eval, replace=False)

    flipped = 0
    for idx in sample_indices:
        original_pred = model.predict(X_arr[idx:idx+1])[0]

        # Correction #8: Threshold-based importance selection
        # Use features with |SHAP| > 50% of the max importance
        abs_sv = np.abs(sv[idx])
        max_importance = abs_sv.max()
        if max_importance > 0:
            importance_threshold = max_importance * 0.5
            important_features = np.where(abs_sv >= importance_threshold)[0]
            # Fall back to top-k if threshold selects too few or too many
            if len(important_features) == 0 or len(important_features) > top_k * 2:
                important_features = np.argsort(abs_sv)[-top_k:]
        else:
            important_features = np.argsort(abs_sv)[-top_k:]

        # Correction #3: Replace with feature medians instead of zero
        perturbed = X_arr[idx:idx+1].copy()
        perturbed[0, important_features] = feature_medians[important_features]

        new_pred = model.predict(perturbed)[0]
        if new_pred != original_pred:
            flipped += 1

    fidelity = flipped / n_eval
    return fidelity