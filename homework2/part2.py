import pandas as pd, matplotlib.pyplot as plt
from statistics import mean
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Conditions
GROUP_NUMBER = 35
k = [1,3,5,9]

data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])
df = df.dropna()
data = df.drop(columns=["Class"]).values
results = df[df.keys()[-1]].astype('string').values

accuracies_test_f, accuracies_train_f, accuracies_test_d, accuracies_train_d = [], [], [], []

kfold = KFold(n_splits=10, shuffle=True, random_state=GROUP_NUMBER)

for i in k:
    clf1 = DecisionTreeClassifier(max_features=i, random_state=GROUP_NUMBER) # 5i.
    clf2 = DecisionTreeClassifier(max_depth=i, random_state=GROUP_NUMBER) # 5ii.

    new_data = SelectKBest(score_func=mutual_info_classif, k=i).fit_transform(data, results)
    scores_f = cross_validate(clf1, new_data, results, scoring="accuracy", return_train_score=True, cv=kfold)
    scores_d = cross_validate(clf2, data, results, scoring="accuracy", return_train_score=True, cv=kfold)

    new_acc_test_f, new_acc_train_f = mean(scores_f['test_score']), mean(scores_f['train_score'])
    new_acc_test_d, new_acc_train_d = mean(scores_d['test_score']), mean(scores_d['train_score'])
    accuracies_test_f.append(new_acc_test_f) ; accuracies_train_f.append(new_acc_train_f)
    accuracies_test_d.append(new_acc_test_d) ; accuracies_train_d.append(new_acc_train_d)

    print("Accuracy: Test-{} # Train-{} for {} features\nAccuracy: Test-{} # Train-{} for {} max depth\n"
        .format(new_acc_test_f, new_acc_train_f, i, new_acc_test_d, new_acc_train_d, i))

# Plot and Save Selected Features plot
test = plt.plot(k, accuracies_test_f, marker='o') ; train = plt.plot(k, accuracies_train_f, marker='o')
plt.xticks(k)
plt.ylabel("Accuracy") ; plt.xlabel("Number of selected features")
plt.legend(["Testing accuracy", "Training accuracy"], loc='lower right')
plt.savefig("graph_f.png")
# Plot and Save Maximum Depth plot
test.pop(0).remove() ; train.pop(0).remove() ; plt.gca().set_prop_cycle(None) # delete old lines
test = plt.plot(k, accuracies_test_d, marker='o') ; train = plt.plot(k, accuracies_train_d, marker='o')
plt.xlabel("Maximum tree depth")
plt.savefig("graph_d.png")