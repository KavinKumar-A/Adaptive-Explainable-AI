from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score, f1_score
)
import numpy as np


def evaluate(model, X_test, y_test):
    """
    Evaluate the model with the full metric suite per Paper §6.4:
    Accuracy, Precision, Recall, F1 Score, FPR, ROC-AUC.
    """
    y_pred = model.predict(X_test)

    # Probabilities (for ROC-AUC)
    try:
        y_prob = model.predict_proba(X_test)
    except AttributeError:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Determine averaging strategy based on class count
    n_classes = len(np.unique(y_test))
    avg = 'binary' if n_classes == 2 else 'weighted'

    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

    # False Positive Rate (Paper §6.4)
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        # Macro-average FPR across all classes
        fpr_per_class = []
        for i in range(n_classes):
            fp_i = cm[:, i].sum() - cm[i, i]
            tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fpr_per_class.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0)
        fpr = np.mean(fpr_per_class)

    # ROC-AUC
    try:
        if y_prob is not None:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        else:
            roc_auc = "N/A"
    except Exception:
        roc_auc = "N/A"

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return acc, metrics