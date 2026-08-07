import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def plot_training_history(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }
    return metrics

def print_metrics(metrics):
    print("\nModel Evaluation\n")
    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")

def plot_confusion_matrix(y_true, y_pred):
    """
    Display confusion matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()

def print_classification_report(y_true, y_pred):
    print(
        classification_report(
            y_true,
            y_pred
        )
    )

def save_results(metrics, filepath):
    results = pd.DataFrame(
        metrics.items(), columns = ["Metric", "Value"]
    )

    results.to_csv(
        filepath,
        index=False
    )

if __name__ == "__main__":
    print("Utility Module loaded Successfully")