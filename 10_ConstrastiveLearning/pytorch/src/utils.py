import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import \
    accuracy_score, recall_score, precision_score, f1_score


def plot_features(model, plot_dataset, labels_dict, device='cuda:0', save_path=None):
    batch_size = 32
    loader = DataLoader(plot_dataset,
        batch_size=batch_size, num_workers=32,
        drop_last=True,
        shuffle=False)

    # Eval
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, desc='Plotting TSNE')):
            x1 = batch['x1'].to(
                device=device,
                dtype=torch.float)
            label = batch['label'].cpu().numpy()

            z1 = model(x1)
            z1 = z1.cpu().data.numpy()

            labels.append(label)
            feats.append(z1)

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    # tsne: reduce dimensionality for easy plotting
    tsne_model = TSNE(n_components=2, perplexity=50)
    feats_2D = tsne_model.fit_transform(feats)

    # plit
    plt.figure(figsize=(10, 7))
    for idx, class_name in labels_dict.items():
        plt.scatter(
            x=feats_2D[labels==idx, 1],
            y=feats_2D[labels==idx, 0],
            s=5, label=f"Class {idx}: {class_name}")

    plt.legend(
        loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    if save_path is not None:
        plt.savefig(save_path,
            dpi=100, bbox_inches='tight', facecolor='white')
    else:
        plt.show()


def compute_metrics(preds, labels, average='macro'):
    assert average in [None, 'micro', 'macro', 'weighted']

    predictions = np.argmax(preds, axis=-1)

    # accuracy, precision, recall, f1-score
    acc = accuracy_score(
        y_true=labels, y_pred=predictions,
        normalize=True)
    r = recall_score(
        y_true=labels, y_pred=predictions,
        average=average, zero_division=0)
    p = precision_score(
        y_true=labels, y_pred=predictions,
        average=average, zero_division=0)
    f1 = f1_score(
        y_true=labels, y_pred=predictions,
        average=average, zero_division=0)

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1 }
