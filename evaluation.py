import numpy as np
from sklearn.model_selection import KFold

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

def evaluate_cf(classifier, data, labels, k_folds: int):
    """
    Perform CV with the calc_accuracy as evaluation criteria
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    acc = []
    for train_index, test_index in kf.split(data):
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        classifier.fit(X_train, y_train)
        y_predictions = classifier.predict(X_test)
        temp_acc = calc_accuracy(y_predictions, y_test)
        print(temp_acc)
        acc.append(temp_acc)
    print(classifier)
    print(f'Estimated accuracy with {k_folds}-CV: {[(key, round(sum([acc[i][key] for i in range(len(acc))])/ len(acc), 2)) for key in acc[0]]}')
    