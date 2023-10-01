import pandas as pd, matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict

data = arff.loadarff('kin8nm.arff')
df = pd.DataFrame(data[0])
df = df.dropna()
data = df.drop(columns=["y"]).values
y = df[df.keys()[-1]].values
y_resid1, y_resid2 = [], []

kfold = KFold(n_splits=5, shuffle=True, random_state=0)

regr1 = MLPRegressor(hidden_layer_sizes=(3,2), max_iter=2000, activation='relu', alpha=2.5, 
                        random_state=0)
regr2 = MLPRegressor(hidden_layer_sizes=(3,2), max_iter=2000, activation='relu', alpha=0, 
                        random_state=0)

y_pred1 = cross_val_predict(regr1, data, y, cv=kfold)
y_pred2 = cross_val_predict(regr2, data, y, cv=kfold)

# residual = actual - expected
y_resid1 = [y-y_pred for y, y_pred in zip(y, y_pred1)]
y_resid2 = [y-y_pred for y, y_pred in zip(y, y_pred2)]

fig = plt.figure(1, figsize=(9,6))
ax = fig.add_subplot(111)
bp = ax.boxplot([y_resid1,y_resid2], showmeans=True, meanline=True, patch_artist=True)
ax.set_xticklabels(['With regularization (alpha=2.5)','Without regularization'])
ax.set_title('Residual plot')

#Customize boxplot
for box in bp['boxes']:
    box.set(color='#7570b3', linewidth=2)
    box.set(facecolor = '#1b9e77' )
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)
for flier in bp['fliers']:
    flier.set(marker='o', color='green', alpha=0.5)
for mean in bp['means']:
    mean.set(linestyle='--', linewidth=2, color='purple')

fig.savefig('residual_plot.png', bbox_inches='tight')
plt.show()