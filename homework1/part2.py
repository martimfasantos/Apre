import pandas as pd
from statistics import mean
from scipy.io import arff
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Conditions
FOLDS = 10
GROUP_NUMBER = 35
k = [3,5,7]
EUCLIDEAN_DIST= 2
WEIGHTS = "uniform"

data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])
df = df.dropna()
data = df.drop(columns=["Class"]).values
results = df[df.keys()[-1]].astype('string').values

cross_val = KFold(n_splits=FOLDS, shuffle=True, random_state=GROUP_NUMBER)

print("#######  EX. 6 #######\n")
for i in k:
    model = KNeighborsClassifier(n_neighbors=i, weights=WEIGHTS, p=EUCLIDEAN_DIST)
    cv_scores = cross_val_score(model, data, results, scoring='accuracy', cv=cross_val)
    print(" - Accuracies (k={}): {} \n\t ## Mean Accuracy: {}\n".format(i,cv_scores,mean(cv_scores)))

print("\n#######  EX. 7 #######\n")
modelKNN = KNeighborsClassifier(n_neighbors=3, weights=WEIGHTS, p=EUCLIDEAN_DIST)
accuracyKNN = cross_val_score(modelKNN, data, results)
modelNB = MultinomialNB()
accuracyNB = cross_val_score(modelNB, data, results)
print("statistic = {} ; p-value = {}".format(ttest_rel(accuracyKNN, accuracyNB, alternative="greater")[0], \
                                            ttest_rel(accuracyKNN, accuracyNB, alternative="greater")[1]))


