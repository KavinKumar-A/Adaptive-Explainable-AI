# AXAI-IDS — Adaptive Explainable AI Intrusion Detection System

A research-grade Machine Learning-based Intrusion Detection System implementing the AXAI-IDS framework. Features Explainable AI (SHAP + LIME) driving an adaptive Zero-Trust policy engine.

## Features

- **Multi-Dataset Support**: CIC-IDS2017, UNSW-NB15 (primary, per paper), NSL-KDD (optional)
- **4 ML Models**: Random Forest, XGBoost, LightGBM, Deep Neural Network (MLP)
- **Explainable AI**: SHAP (global) + LIME (local) with fidelity & latency metrics
- **Adaptive Policy Engine**: 5-tier XAI-driven zero-trust enforcement
- **Interactive Dashboard**: Streamlit-based monitoring with model comparison
- **Human Feedback Loop**: Analyst corrections adapt policy thresholds

## Project Structure

```
RESEARCH_PROJECT/
├── datasets/
│   ├── CIC_IDS2017/          # CIC-IDS2017 dataset files
│   ├── NSL_KDD/              # NSL-KDD dataset files
│   └── UNSW_NB15/            # UNSW-NB15 dataset files
├── src/
│   ├── data_loader/          # Dataset loading modules
│   ├── preprocessing/        # Cleaning, encoding, MinMax normalization
│   ├── features/             # Correlation pruning + ANOVA feature selection
│   ├── models/               # RF, XGBoost, LightGBM, DNN implementations
│   ├── explainability/       # SHAP (fidelity + latency) & LIME explainers
│   ├── policy_engine/        # 5-tier adaptive policy + feedback handler
│   └── dashboard/            # Streamlit dashboard
├── models/                   # Trained model artifacts (per dataset)
├── reports/                  # Results, figures, SHAP/LIME outputs
├── main.py                   # Main comparative pipeline
└── requirements.txt          # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train All Models
```bash
python main.py
```
Processes CIC-IDS2017 + UNSW-NB15 × 4 models (RF, XGBoost, LightGBM, DNN).

### Launch Dashboard
```bash
streamlit run src/dashboard/dashboard.py
```

## ML Models (Paper §6.2)

| Model | Key Advantage |
|-------|--------------|
| Random Forest | Robust baseline, inherent feature importance |
| XGBoost | High accuracy, L1/L2 regularization |
| LightGBM | Fastest training, low memory (leaf-wise growth) |
| DNN (MLP) | Non-linear feature abstractions, 3 hidden layers |

## Evaluation Metrics (Paper §6.4)

- **Accuracy, Precision, Recall, F1 Score** — Classification performance
- **False Positive Rate (FPR)** — Operational disruption measurement
- **ROC-AUC** — Discrimination capability
- **Explanation Fidelity** — XAI accuracy (Top-3 feature perturbation)
- **XAI Latency** — SHAP/LIME computation time per flow

## Adaptive Policy Engine (Paper §5.1.5)

| Risk Score | Action |
|-----------|--------|
| > 0.95 | Isolate Subnet |
| > 0.80 | Block Connection |
| > 0.50 | Require MFA |
| > 0.30 | Rate Limit |
| ≤ 0.30 | Allow with Monitoring |

XAI-driven micro-policies map specific anomalous features to targeted actions (e.g., anomalous DNS → rate-limit DNS only).

## Preprocessing Pipeline (Paper §7.1–§7.2)

1. **Data Cleansing**: Inf→NaN, median imputation, deduplication
2. **Label Encoding**: Categorical → numeric via LabelEncoder
3. **MinMax Normalization**: Features scaled to [0, 1]
4. **Feature Selection**: Correlation pruning (Pearson > 0.90) + ANOVA F-test → Top-20

## Results

Results are saved to `reports/results.txt` and `reports/results.json` after running `python main.py`.

## Citation

```
@misc{axai_ids_2026,
  title={Adaptive Explainable AI Framework for Autonomous Intrusion Detection and Policy Enforcement in Zero-Trust Networks},
  author={Research Team},
  year={2026}
}
```