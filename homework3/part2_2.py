import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPClassifier

data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])
df = df.dropna()
data = df.drop(columns=["Class"]).values
results = df[df.keys()[-1]].astype('string').values

kfold = KFold(n_splits=5, shuffle=True, random_state=0)

class1 = MLPClassifier(hidden_layer_sizes=(3,2), max_iter=2000, activation='relu', alpha=1,
                        random_state=0, early_stopping=False)
class2 = MLPClassifier(hidden_layer_sizes=(3,2), max_iter=2000, activation='relu', alpha=1,
                        random_state=0, early_stopping=True)

y_pred1 = cross_val_predict(class1, data, results, cv=kfold)
y_pred2 = cross_val_predict(class2, data, results, cv=kfold)
cm1 = confusion_matrix(results, y_pred1)        ;       cm2 = confusion_matrix(results, y_pred2)
tn1, fp1, fn1, tp1 = cm1.ravel()                ;       tn2, fp2, fn2, tp2 = cm2.ravel()

print("Without early stopping:\n{}\nTN:{} FP:{} FN:{} TP:{}\nWith early stopping:\n{}\nTN:{} FP:{} FN:{} TP:{}"
      .format(cm1, tn1, fp1, fn1, tp1, cm2, tn2, fp2, fn2, tp2))

ConfusionMatrixDisplay.from_predictions(results, y_pred1, display_labels=['benign','malignant'])
ConfusionMatrixDisplay.from_predictions(results, y_pred2, display_labels=['benign','malignant'])
plt.show()
