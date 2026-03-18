# AXAI-IDS: Project Execution Guide

This guide provides step-by-step instructions on how to set up the environment, install the required dependencies, and run the AXAI-IDS (Adaptive Explainable AI Intrusion Detection System) framework on macOS/Linux.

---

## 1. Prerequisites

Ensure you have Python 3.9+ installed on your system. You can verify your Python version by running:
```bash
python3 --version
```

---

## 2. Set Up the Virtual Environment (venv)

It is highly recommended to run this project inside a Python virtual environment to prevent dependency conflicts with other projects.

**Step 2.1: Open your terminal and navigate to the project directory**
```bash
cd "/Users/kavinkumara/Hackathon/Research/research project"
```

**Step 2.2: Create the virtual environment**
Run the following command to create a virtual environment named `.venv` in the root of the project:
```bash
python3 -m venv .venv
```

**Step 2.3: Activate the virtual environment**
To access and activate the virtual environment, run:
```bash
source .venv/bin/activate
```
*(You should now see `(.venv)` at the beginning of your terminal prompt, indicating that the environment is active).*

---

## 3. Install Required Libraries

With the virtual environment active, you need to install the project dependencies listed in `requirements.txt`.

**Step 3.1: Upgrade pip (optional but recommended)**
```bash
pip install --upgrade pip
```

**Step 3.2: Install the dependencies**
```bash
pip install -r requirements.txt
```
This will install all necessary libraries including `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `lime`, and `streamlit` cleanly inside your `.venv`.

---

## 4. Run the Core ML Pipeline

Once the dependencies are installed, you can run the main training and evaluation pipeline. This pipeline loads the datasets, preprocesses the data, trains the models (RF, XGBoost, LightGBM, DNN), and evaluates their performance.

Run the main execution script:
```bash
python3 main.py
```
*(Note: Ensure your `datasets/` folder contains the required CIC-IDS2017 and UNSW-NB15 CSV files before running).*

---

## 5. Generate Academic Results & Figures

To generate the high-quality IEEE figures (PNGs) and the results data required for the research paper:

```bash
python3 generate_results.py
```
This script will output the evaluation matrices and save all the charts (like the SHAP latency comparison, Policy Disruption graphs, and Model Performance graphs) into the `reports/figures/` directory.

---

## 6. Compile the LaTeX Paper

To convert the Markdown paper draft (`AXAI_IDS_Baseline_Paper.md`) into a fully formatted IEEE LateX document (`AXAI_IDS_Paper.tex`) ready for compilation (e.g., Overleaf):

```bash
python3 create_tex.py
```

---

## 7. Run the Security Dashboard (Streamlit)

The project includes an interactive web dashboard for SOC analysts to view traffic classifications, XAI explanations, and policy enforcement actions in real-time.

To launch the dashboard, use Streamlit:
```bash
streamlit run src/dashboard/dashboard.py
```
This will start a local web server, and the dashboard should automatically open in your default web browser (usually at `http://localhost:8501`).

### How to Use the Dashboard:
1. **Select a Model**: On the sidebar, use the dropdown menu to choose which embedded machine learning model you want to evaluate (Random Forest, XGBoost, LightGBM, or Deep Neural Network).
2. **Review Metrics**: The top of the main page will display the core evaluation metrics for the selected model: Accuracy, False Positive Rate (FPR%), and the Disruption Reduction percentage achieved by the policy engine.
3. **Analyze Flow Data**: In the center section, you can select specific network flows from the evaluation set to inspect. 
4. **Trigger XAI & Policy Engine**: When you select a flow flagged as anomalous ("Attack"), the dashboard will simultaneously display:
   - The **SHAP/LIME Feature Importance diagram**, visually calculating *why* the flow was flagged (e.g., highlighting that `Flow Duration` was abnormally high).
   - The **Autonomous Policy Action**, demonstrating how the policy engine consumed the XAI data to create a granular mitigation rule (like "Rate Limit specific protocol") instead of broadly blocking the IP address.

---

## 8. Deactivate the Virtual Environment

When you are finished working on the project, you can exit the virtual environment by simply typing:
```bash
deactivate
```
This will return your terminal to its normal system Python state.
