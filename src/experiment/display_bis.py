import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def show_auc(clf, X, y, name=None, plot=True):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X)
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        if plot:
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
    else:
        ns_proba = [0 for _ in range(len(y))]
        y_proba = clf.predict_proba(X)
        y_proba = y_proba[:, 1]
        # calculate scores
        ns_auc = roc_auc_score(y, ns_proba)
        lr_auc = roc_auc_score(y, y_proba)
        if not plot:
            return lr_auc * 100
        # summarize scores
        #print('No Skill: ROC AUC=%.3f' % (ns_auc))
        #print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y, ns_proba)
        lr_fpr, lr_tpr, _ = roc_curve(y, y_proba)
        # plot the roc curve for the model
        if name is None:
            plt.title(f'Receiver operating characteristic')
        else:
            plt.title(f'Receiver operating characteristic for {name}')
        plt.plot(ns_fpr, ns_tpr, linestyle='--')
        plt.plot(lr_fpr, lr_tpr, marker='.', label=f'ROC curve (area = {lr_auc*100:.2f}')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

        return lr_auc * 100



    