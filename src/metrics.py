import numpy as np

def Accuracy(preds, labels):
    return np.mean(preds == labels)

def Precision(preds, labels):
    classes = np.unique(labels)
    precisions = []
    for cls in classes:
        tp = np.sum((preds == cls) & (labels == cls))
        fp = np.sum((preds == cls) & (labels != cls))
        precisions.append(tp / (tp + fp + 1e-7))
    return np.mean(precisions)

def Recall(preds, labels):
    classes = np.unique(labels)
    recalls = []
    for cls in classes:
        tp = np.sum((preds == cls) & (labels == cls))
        fn = np.sum((preds != cls) & (labels == cls))
        recalls.append(tp / (tp + fn + 1e-7))
    return np.mean(recalls)