import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def show_auc(lr, X, y, name=None):
    y_score = lr.decision_function(X)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr['micro'], tpr['micro'], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % (roc_auc['micro'] * 100))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if name is None:
        plt.title('Receiver operating characteristic')
    else:
        plt.title('Receiver operating characteristic for {}'.format(name))
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc['micro'] * 100