from src.data_loader.dataset_selector import load_dataset
from src.preprocessing.clean_data import clean
from src.preprocessing.encode_features import encode
from src.preprocessing.normalize_data import normalize
from src.preprocessing.split_dataset import split
from src.features.feature_selection import select_features
from src.models.train_model import train
from src.models.evaluate_model import evaluate
from src.policy_engine.policy_rules import get_rules
from src.policy_engine.decision_engine import decide, simulate_policy_disruption
import pickle
import os
import json
import numpy as np
import warnings

# ──────────── Model Registry (Paper §6.2) ────────────
# Suppress global scikit-learn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
MODEL_REGISTRY = {
    "RandomForest": "src.models.random_forest",
    "XGBoost":      "src.models.xgboost_model",
    "LightGBM":     "src.models.lightgbm_model",
    "DNN":          "src.models.dnn_model",
}

# Datasets as specified in Paper §6.1 (CIC-IDS2017 + UNSW-NB15 + NSL-KDD)
DATASETS = ["cicids", "unsw", "nslkdd"]

# Sampling cap for tractable dev/test runs
SAMPLE_SIZE = 10000


def import_model_getter(module_path):
    """Dynamically import get_model() from a model module."""
    import importlib
    mod = importlib.import_module(module_path)
    return mod.get_model


def main():
    """
    AXAI-IDS Main Pipeline (Paper §6 & §7).
    Loops all models × all datasets, generating comparative results.
    """
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        try:
            # ── Phase 1: Load Dataset (Paper §6.1) ──
            print(f"[Phase 1] Loading {ds_name} dataset...")
            data = load_dataset(ds_name)
            if len(data) > SAMPLE_SIZE:
                data = data.sample(SAMPLE_SIZE, random_state=42)
            print(f"  Loaded {data.shape[0]} rows × {data.shape[1]} cols")

            # ── Phase 2: Preprocessing (Paper §7.1) ──
            print("[Phase 2] Preprocessing...")
            data = clean(data)    # Median imputation + inf handling
            data = encode(data)   # Label encoding

            # Identify target column
            target_col = None
            for col in ['label', 'Label', 'attack_type', 'class']:
                if col in data.columns:
                    target_col = col
                    break
            if not target_col:
                print(f"  ERROR: Target column not found for {ds_name}. Skipping.")
                continue

            # MinMax normalization [0, 1] (Paper §7.1)
            data, scaler = normalize(data, target_col)

            # Stratified split (Paper §7.3)
            X_train, X_test, y_train, y_test = split(data, target_col)
            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

            # ── Phase 3: Feature Selection (Paper §7.2) ──
            print("[Phase 3] Feature selection (correlation pruning + ANOVA)...")
            X_train_sel, selector, kept_indices = select_features(X_train, y_train, k=20)

            # Apply same selection to test set
            if hasattr(X_test, 'iloc'):
                X_test_pruned = X_test.iloc[:, kept_indices]
            else:
                X_test_pruned = X_test[:, kept_indices]
            X_test_sel = selector.transform(X_test_pruned)
            print(f"  After selection: {X_train_sel.shape[1]} features")

            # Get feature names for XAI
            if hasattr(X_train, 'columns'):
                pruned_names = [X_train.columns[i] for i in kept_indices]
                selected_mask = selector.get_support()
                selected_names = [pruned_names[i] for i, s in enumerate(selected_mask) if s]
            else:
                selected_names = [f"feature_{i}" for i in range(X_train_sel.shape[1])]

            # Save preprocessing artifacts
            os.makedirs(f'models/{ds_name}', exist_ok=True)
            with open(f'models/{ds_name}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open(f'models/{ds_name}/selector.pkl', 'wb') as f:
                pickle.dump(selector, f)

            ds_results = {}

            # ── Phase 4-7: Train & Evaluate Each Model ──
            for model_name, module_path in MODEL_REGISTRY.items():
                print(f"\n  ── Model: {model_name} ──")

                try:
                    # Phase 4: Model Training (Paper §7.3)
                    model_path = f'models/{ds_name}/{model_name}_model.pkl'
                    if os.path.exists(model_path):
                        print(f"  [Phase 4] Loading pre-trained {model_name}...")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    else:
                        print(f"  [Phase 4] Training {model_name}...")
                        get_model = import_model_getter(module_path)
                        model = get_model()
                        model = train(model, X_train_sel, y_train)
                        # Save model
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)

                    # Phase 4b: Evaluation (Paper §6.4)
                    acc, metrics = evaluate(model, X_test_sel, y_test)
                    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                    print(f"    Precision: {metrics['precision']:.4f}")
                    print(f"    Recall:    {metrics['recall']:.4f}")
                    print(f"    F1 Score:  {metrics['f1_score']:.4f}")
                    print(f"    FPR:       {metrics['false_positive_rate']:.4f}" if isinstance(metrics['false_positive_rate'], float) else f"    FPR: {metrics['false_positive_rate']}")

                    # Phase 5: SHAP Explainability (Paper §7.4)
                    print(f"  [Phase 5] SHAP explanations for {model_name}...")
                    from src.explainability.shap_explainer import explain as shap_explain
                    xai_subset = X_test_sel[:100]
                    shap_results = shap_explain(model, xai_subset, ds_name=ds_name, model_name=model_name)
                    print(f"    SHAP latency/flow: {shap_results['latency_per_flow_ms']:.2f} ms")
                    print(f"    Explanation fidelity: {shap_results['explanation_fidelity']:.4f}")

                    # Phase 5b: LIME Explanations (Paper §6.3)
                    print(f"  [Phase 5b] LIME explanation for {model_name}...")
                    from src.explainability.lime_explainer import explain as lime_explain
                    lime_results = lime_explain(
                        model, X_train_sel, X_test_sel,
                        instance_idx=0, ds_name=ds_name, model_name=model_name
                    )
                    print(f"    LIME latency: {lime_results['latency_ms']:.2f} ms")

                    # Phase 6: Policy Engine Simulation (Paper §7.5)
                    print(f"  [Phase 6] Policy engine simulation...")
                    rules = get_rules()
                    sample_prob = float(model.predict_proba(X_test_sel[:1])[0][1]) if hasattr(model, 'predict_proba') else 0.5

                    # Get SHAP values for the sample
                    sv = shap_results['shap_values']
                    if isinstance(sv, list):
                        sv_sample = np.array(sv[1][0]) if len(sv) > 1 else np.array(sv[0][0])
                    else:
                        sv = np.array(sv)
                        if sv.ndim == 3:
                            # Shape (n_samples, n_features, n_classes)
                            n_classes = sv.shape[2]
                            target_class = 1 if n_classes > 1 else 0
                            sv_sample = sv[0, :, target_class]
                        else:
                            sv_sample = sv[0]

                    decision = decide(sample_prob, rules, sv_sample, selected_names)
                    disruption = simulate_policy_disruption(
                        decision['base_action'], decision['targeted_actions']
                    )
                    print(f"    Base action: {decision['base_action']}")
                    print(f"    Targeted actions: {len(decision['targeted_actions'])}")
                    print(f"    Disruption reduction: {disruption['disruption_reduction_pct']:.1f}%")

                    # Store results
                    ds_results[model_name] = {
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'false_positive_rate': metrics['false_positive_rate'] if isinstance(metrics['false_positive_rate'], float) else 0,
                        'roc_auc': metrics['roc_auc'] if isinstance(metrics['roc_auc'], float) else 0,
                        'shap_latency_per_flow_ms': shap_results['latency_per_flow_ms'],
                        'lime_latency_ms': lime_results['latency_ms'],
                        'explanation_fidelity': shap_results['explanation_fidelity'],
                        'disruption_reduction_pct': disruption['disruption_reduction_pct'],
                    }

                except Exception as e:
                    print(f"    ERROR with {model_name}: {str(e)}")
                    ds_results[model_name] = {'error': str(e)}
                    continue

            all_results[ds_name] = ds_results

        except Exception as e:
            print(f"ERROR processing {ds_name}: {str(e)}")
            continue

    # ── Save Comprehensive Results ──
    save_results(all_results)
    print(f"\n{'='*60}")
    print("  All datasets and models processed!")
    print(f"  Results saved to reports/results.txt")
    print(f"{'='*60}")


def save_results(all_results):
    """Save comprehensive results report per Paper §8."""
    with open('reports/results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  AXAI-IDS Framework — Comprehensive Results\n")
        f.write("=" * 70 + "\n\n")

        for ds_name, models in all_results.items():
            f.write(f"\nDataset: {ds_name.upper()}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Model':<18} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FPR':>8} {'Fidelity':>10}\n")
            f.write("-" * 50 + "\n")

            for model_name, metrics in models.items():
                if 'error' in metrics:
                    f.write(f"{model_name:<18} ERROR: {metrics['error']}\n")
                else:
                    f.write(
                        f"{model_name:<18} "
                        f"{metrics['accuracy']:>7.4f} "
                        f"{metrics['precision']:>7.4f} "
                        f"{metrics['recall']:>7.4f} "
                        f"{metrics['f1_score']:>7.4f} "
                        f"{metrics['false_positive_rate']:>7.4f} "
                        f"{metrics['explanation_fidelity']:>9.4f}\n"
                    )

            f.write("\n  XAI Latency & Policy Simulation:\n")
            for model_name, metrics in models.items():
                if 'error' not in metrics:
                    f.write(
                        f"    {model_name}: SHAP {metrics['shap_latency_per_flow_ms']:.2f}ms/flow, "
                        f"LIME {metrics['lime_latency_ms']:.2f}ms, "
                        f"Disruption Reduction {metrics['disruption_reduction_pct']:.1f}%\n"
                    )
            f.write("\n")

    # Also save as JSON for programmatic access
    serializable = {}
    for ds, models in all_results.items():
        serializable[ds] = {}
        for m, v in models.items():
            serializable[ds][m] = {k: (float(vv) if isinstance(vv, (np.floating, float)) else vv) for k, vv in v.items()}

    with open('reports/results.json', 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


if __name__ == "__main__":
    main()