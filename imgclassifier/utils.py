from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def evaluation_report(targets, preds, average='macro'):

    pr = precision_score(y_true=targets, y_pred=preds, average=average)
    re = recall_score(y_true=targets, y_pred=preds, average=average)
    f1 = f1_score(y_true=targets, y_pred=preds, average=average)
    acc = accuracy_score(y_true=targets, y_pred=preds)

    print(f'Precision = {pr:.4f}, \nRecall = {re:.4f}, \nF1 Score = {f1:.4f}, \nAccuracy Score = {acc:.4f}')
