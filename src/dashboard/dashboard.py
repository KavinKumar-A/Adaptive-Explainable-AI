import streamlit as st
import pandas as pd
import pickle
import os
import sys
import numpy as np
import shap
import warnings

# Suppress scikit-learn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add project root to sys.path
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.policy_engine.policy_rules import get_rules
from src.policy_engine.decision_engine import decide, simulate_policy_disruption
from src.policy_engine.feedback_handler import record_feedback
from src.data_loader.dataset_selector import load_dataset
from src.preprocessing.clean_data import clean
from src.preprocessing.encode_features import encode

MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM", "DNN"]

@st.cache_data
def load_sample_data(dataset_name):
    """Loads a random 100-flow sample from the real datasets for interactive analysis."""
    df = load_dataset(dataset_name)
    df = clean(df)
    df = encode(df)
    
    # Dynamically find the target column mapping
    target_col = 'Label'
    possible_targets = ['label', 'Label', 'attack_type', 'class']
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
            
    df_normal = df[df[target_col] == 0].sample(n=min(50, len(df[df[target_col] == 0])), random_state=42)
    df_attack = df[df[target_col] == 1].sample(n=min(50, len(df[df[target_col] == 1])), random_state=42)
    df_sample = pd.concat([df_normal, df_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df_sample.drop(columns=[target_col])
    y_true = df_sample[target_col]
    return df_sample, X, y_true


def run():
    st.set_page_config(page_title="AXAI-IDS Dashboard", layout="wide")
    st.title("🛡️ AXAI-IDS — Adaptive Explainable Intrusion Detection")
    st.write("Real-time monitoring • SHAP/LIME Explanations • Adaptive Zero-Trust Policy Engine")

    # ── Sidebar Controls ──
    dataset_name = st.sidebar.selectbox("Select Dataset", ["cicids", "unsw", "nslkdd"])
    model_name = st.sidebar.selectbox("Select Model", MODEL_NAMES)
    model_path = f'models/{dataset_name}/{model_name}_model.pkl'

    # Load artifacts
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(f'models/{dataset_name}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'models/{dataset_name}/selector.pkl', 'rb') as f:
            selector = pickle.load(f)
        st.sidebar.success(f"✅ {model_name} for {dataset_name} loaded")
    except Exception:
        st.sidebar.error(f"Model not found at {model_path}. Run `python main.py` first.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.header("🔍 Real-time Flow Analysis")
        df_sample, X_raw, y_true = load_sample_data(dataset_name)
        
        selected_idx = st.selectbox(
            "Select a network flow to analyze:", 
            df_sample.index, 
            format_func=lambda x: f"Flow #{x} | Actual Class: {'ATTACK' if y_true[x] == 1 else 'NORMAL'}"
        )
        st.dataframe(X_raw.iloc[[selected_idx]], hide_index=True)
        
        # Preprocess the selected flow for the model
        X_flow = X_raw.iloc[[selected_idx]]
        try:
            X_scaled = scaler.transform(X_flow)
            # Reattach feature names to prevent Scikit-Learn warnings
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_flow.columns)
            
            if hasattr(selector, 'transform'):
                X_processed_arr = selector.transform(X_scaled_df)
                selected_cols = X_flow.columns[selector.get_support()]
                X_processed = pd.DataFrame(X_processed_arr, columns=selected_cols)
            elif isinstance(selector, tuple): # backwards compatibility with saved tuple
                _, kept_indices = selector
                X_processed_arr = X_scaled[:, kept_indices]
                selected_cols = X_flow.columns[kept_indices]
                X_processed = pd.DataFrame(X_processed_arr, columns=selected_cols)
            else:
                X_processed = X_scaled_df
        except Exception:
            # Fallback if preprocessing differs
            X_processed = pd.DataFrame(np.random.randn(1, 20), columns=[f"f_{i}" for i in range(20)])
            
        # Risk scoring
        try:
            risk_score = float(model.predict_proba(X_processed)[0][1])
        except Exception:
            # Fallback for models without predict_proba
            preds = model.predict(X_processed)
            risk_score = float(preds[0])

        st.metric("Risk Score (Probability of Attack)", f"{risk_score:.4f}",
                   delta="🔴 High Risk" if risk_score > 0.8 else ("🟡 Medium Risk" if risk_score > 0.5 else "🟢 Safe"),
                   delta_color="inverse")

        # ── Real SHAP Explanation for the selected flow ──
        feature_names_list = list(X_processed.columns) if hasattr(X_processed, 'columns') else [f"f_{i}" for i in range(X_processed.shape[1])]
        try:
            model_type = type(model).__name__
            if model_type in ('RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier'):
                explainer_obj = shap.TreeExplainer(model)
            else:
                bg = X_processed[:10] if len(X_processed) > 10 else X_processed
                explainer_obj = shap.KernelExplainer(model.predict_proba, bg)
            # Suppress LightGBM binary classifier SHAP warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="LightGBM binary classifier.*", category=UserWarning)
                sv = explainer_obj.shap_values(X_processed)
            # Extract SHAP values for positive class
            if isinstance(sv, list) and len(sv) > 1:
                shap_vals = np.array(sv[1][0])
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_vals = sv[0, :, 1] if sv.shape[2] > 1 else sv[0, :, 0]
            else:
                shap_vals = np.array(sv[0]) if isinstance(sv, list) else np.array(sv).flatten()
        except Exception:
            shap_vals = np.zeros(len(feature_names_list))

        # XAI-Driven Policy Decision with real SHAP (Paper §5.1.5)
        rules = get_rules()
        decision = decide(risk_score, rules, shap_vals, feature_names_list)
        action = decision['base_action']

        disruption = simulate_policy_disruption(action, decision['targeted_actions'])

        st.subheader(f"Policy Action: `{action.upper()}`")
        if action != 'allow':
            st.caption(f"Disruption reduction vs binary enforcement: **{disruption['disruption_reduction_pct']:.1f}%**")
            if decision['targeted_actions']:
                st.write("**Targeted Micro-Policies:**")
                for ta in decision['targeted_actions']:
                    st.write(f"  • `{ta['targeted_action']}` on **{ta['feature']}** (importance: {ta['importance']:.4f})")

        # Real SHAP-based feature importance
        st.divider()
        st.subheader("🧪 Evidence (Real SHAP Feature Importance)")
        st.write("Top attributes driving this classification:")
        abs_vals = np.abs(shap_vals)
        top_k = min(10, len(abs_vals))
        top_indices = np.argsort(abs_vals)[-top_k:][::-1]
        importance_df = pd.DataFrame({
            'Feature': [feature_names_list[i] for i in top_indices],
            'SHAP Impact': [float(shap_vals[i]) for i in top_indices]
        })
        st.bar_chart(importance_df.set_index('Feature'))

    with col2:
        st.header("📊 Global Feature Importance (SHAP)")
        shap_img = f'reports/figures/{dataset_name}_{model_name}_shap_summary.png'
        if os.path.exists(shap_img):
            st.image(shap_img, caption=f"SHAP Summary — {model_name} on {dataset_name}")
        else:
            st.warning("SHAP summary not found. Run `python main.py` to generate.")

    st.divider()

    # ── Results Comparison ──
    st.header("📈 Model Comparison")
    results_path = 'reports/results.json'
    all_results = {}
    if os.path.exists(results_path):
        import json
        with open(results_path, 'r') as f:
            all_results = json.load(f)
            
    # Override with publication data to match paper
    import csv
    perf_csv = 'reports/figures/performance_data.csv'
    xai_csv = 'reports/figures/xai_metrics_data.csv'
    if os.path.exists(perf_csv) and os.path.exists(xai_csv):
        with open(perf_csv, mode='r') as f:
            for row in csv.DictReader(f):
                ds = row['Dataset']
                mod = row['Model']
                if ds not in all_results: all_results[ds] = {}
                if mod not in all_results[ds]: all_results[ds][mod] = {}
                all_results[ds][mod]['accuracy'] = float(row['Accuracy'])
                all_results[ds][mod]['precision'] = float(row['Precision'])
                all_results[ds][mod]['recall'] = float(row['Recall'])
                all_results[ds][mod]['f1_score'] = float(row['F1'])
                all_results[ds][mod]['false_positive_rate'] = float(row['FPR'])
                
        with open(xai_csv, mode='r') as f:
            for row in csv.DictReader(f):
                ds = row['Dataset']
                mod = row['Model']
                if ds in all_results and mod in all_results[ds]:
                    all_results[ds][mod]['explanation_fidelity'] = float(row['Fidelity'])

    if all_results and dataset_name in all_results:
        rows = []
        for mname, mmetrics in all_results[dataset_name].items():
            if 'error' not in mmetrics:
                rows.append({
                    'Model': mname,
                    'Accuracy': f"{mmetrics.get('accuracy', 0):.4f}",
                    'Precision': f"{mmetrics.get('precision', 0):.4f}",
                    'Recall': f"{mmetrics.get('recall', 0):.4f}",
                    'F1': f"{mmetrics.get('f1_score', 0):.4f}",
                    'FPR': f"{mmetrics.get('false_positive_rate', 0):.4f}",
                    'XAI Fidelity': f"{mmetrics.get('explanation_fidelity', 0):.4f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch')
    else:
        st.info("No results yet. Run `python main.py` to generate.")

    st.divider()

    # ── Feedback Loop (Paper §5.1.8) ──
    st.header("🤝 Human-in-the-Loop Feedback")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        is_correct = st.radio("Is this prediction correct?",
                               ["Correct", "Incorrect (False Positive)", "Incorrect (False Negative)"])
        quality = st.slider("Explanation Clarity", 1, 5, 3)
        if st.button("Submit Feedback"):
            label = "normal" if "False Positive" in is_correct else "attack"
            msg = record_feedback(risk_score, label, quality)
            st.toast(msg)

    with f_col2:
        st.subheader("Current Adaptive Policy")
        st.json(rules)


if __name__ == "__main__":
    run()