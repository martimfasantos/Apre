import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.io import arff

data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])
df = df.dropna()

#separating benigns and malignant lists
grouped = df.groupby(['Class'])
benign = grouped.get_group(b'benign')
malignant = grouped.get_group(b'malignant')

test = pd.DataFrame(np.random.randn(30,9), columns=map(str, range(9)))
fig = plt.figure(figsize=(16,12))

for i in range(9):
    plt.subplot(3,3,i+1)
    bins = np.linspace(1, 11, num = 11)
    weights_benign = np.ones_like(benign.iloc[:,i]) / (len(benign.iloc[:,i]))
    plt.hist(benign.iloc[:,i].values, bins, align="left", alpha = 0.5, label = 'benign', weights=weights_benign)
    weights_malignant = np.ones_like(malignant.iloc[:,i]) / (len(malignant.iloc[:,i]))
    plt.hist(malignant.iloc[:,i].values, bins, align="left", alpha = 0.5, label = 'malignant', weights=weights_malignant)
    plt.xticks(range(1,11))
    plt.legend(loc = 'upper center')
    plt.title("P( {} | Class = ? )".format(df.columns[i]))

fig.tight_layout()  # Improves appearance a bit.
fig.savefig("graphs.jpg")
