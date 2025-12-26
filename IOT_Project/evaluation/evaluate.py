from evaluation.metrics import evaluate_classification
from evaluation.plots import plot_roc_curve, plot_confusion_matrix

def run_evaluation(y_test, y_pred, y_prob):
    metrics = evaluate_classification(y_test, y_pred, y_prob)

    plot_roc_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    
    return metrics
