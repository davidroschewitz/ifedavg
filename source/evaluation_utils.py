from sklearn.metrics import (
    f1_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    average_precision_score,
    roc_auc_score,
)


def get_metric(metric, true, pred, pred_probas):
    if metric == "f1":
        return f1_score(true, pred, zero_division=0, average="weighted")
    elif metric == "recall":
        return recall_score(true, pred, zero_division=0, average="weighted")
    elif metric == "balanced_accuracy":
        return balanced_accuracy_score(true, pred)
    elif metric == "accuracy":
        return accuracy_score(true, pred)
    elif metric == "precision":
        return precision_score(true, pred, zero_division=0, average="weighted")
    elif metric == "pr_auc":
        try:
            if pred_probas.shape[1] == 2:
                return average_precision_score(true, pred_probas[:, 1])
            else:
                return 0.0
        except:
            return 0.0
    elif metric == "roc_auc":
        try:
            if pred_probas.shape[1] == 2:
                return roc_auc_score(true, pred_probas[:, 1])
            else:
                try:
                    return roc_auc_score(true, pred_probas, multi_class="ovo")
                except:
                    return 0.0
        except:
            return 0.0
    else:
        raise NotImplementedError("Metric", metric, "is not yet supported...")
