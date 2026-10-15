import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["QCD", "Top Quark"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Jet Tagging Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (Background Efficiency)")
    ax.set_ylabel("True Positive Rate (Signal Efficiency)")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_latent_space_tsne(
    latent_features: np.ndarray, labels: np.ndarray
) -> plt.Figure:
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(latent_features)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        s=15,
    )
    ax.set_title("t-SNE Latent Space Representation")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Label (0: QCD, 1: Top)")
    plt.tight_layout()
    return fig
