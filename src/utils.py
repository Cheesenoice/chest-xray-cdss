import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_metrics(y_true, y_pred, y_prob, num_classes=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Calculate AUC on classes actually present in y_true
    present = np.unique(y_true)
    try:
        if len(present) == 2:
            auc = roc_auc_score(y_true, y_prob[:, present[1]])
        else:
            auc = roc_auc_score(y_true, y_prob[:, present], multi_class="ovr", average="macro")
    except Exception as e:
        print(f"[WARN] AUC computation failed: {e}")
        auc = 0.0

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "auc_macro": float(auc)
    }
