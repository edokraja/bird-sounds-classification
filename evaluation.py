import numpy as np

def calc_accuracy(predictions: np.ndarray, true_labels: np.ndarray, classes=7):
    """
    Calculates the overall accuracy and the accuracy for each class.
    returns: dict containing accuracies
    """
    acc = {}
    acc["Overall"] = round(sum((predictions-true_labels)==0)/len(true_labels)*100,2)
    for i in range(0, classes):
        pred = predictions[true_labels==i]
        if len(pred) == 0 :
            acc[f"Class {i}"] = 0
        else:
            acc[f"Class {i}"] = round(len(pred[pred==i])/len(pred)*100,2)
    return acc