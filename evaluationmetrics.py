from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def precision_recall_curve_plot(y_test, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

def confusion_matrix_plot(y_test, y_score):
    cm = confusion_matrix(y_test, y_score)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Predicted 0s', 'Predicted 1s'])
    plt.yticks(tick_marks, ['Actual 0s', 'Actual 1s'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
