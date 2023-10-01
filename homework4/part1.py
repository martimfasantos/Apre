import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from numpy import transpose, matmul
import numpy.linalg as la

# Data
x1, x2, x3, x4 = (2,4), (-1,-4), (-1,2), (4,0)
x = [x1,x2,x3,x4]
y1 = [x1[0], x2[0], x3[0], x4[0]]
y2 = [x1[1], x2[1], x3[1], x4[1]]

m1 = x1
m2 = x2
cov1 = [[1,0],[0,1]]
cov2 = [[2,0],[0,2]]

pi1 = 0.7
pi2 = 0.3

print('####### LIKELIHOODS: ######')
for i in x:
    print(f'{i}')
    print(f"c1=1: {multivariate_normal(mean=m1, cov=cov1).pdf(i)}")
    print(f"c2=1: {multivariate_normal(mean=m2, cov=cov2).pdf(i)}")
    print('----')

norm_c1, norm_c2 = [], []
print('####### JOINTS AND PROBS: ######')
for i in x:
    print(f'{i}')
    print(f"c1=1: {pi1*multivariate_normal(mean=m1, cov=cov1).pdf(i)}")
    print(f"c2=2: {pi2*multivariate_normal(mean=m2, cov=cov2).pdf(i)}")
    print(f"sum: {pi1*multivariate_normal(mean=m1, cov=cov1).pdf(i)+pi2*multivariate_normal(mean=m2, cov=cov2).pdf(i)}")
    norm_c1.append(pi1*multivariate_normal(mean=m1, cov=cov1).pdf(i)/(pi1*multivariate_normal(mean=m1, cov=cov1).pdf(i)+pi2*multivariate_normal(mean=m2, cov=cov2).pdf(i)))
    norm_c2.append(pi2*multivariate_normal(mean=m2, cov=cov2).pdf(i)/(pi1*multivariate_normal(mean=m1, cov=cov1).pdf(i)+pi2*multivariate_normal(mean=m2, cov=cov2).pdf(i)))
    print(f"normalizing (c1=1): {norm_c1[-1]}")
    print(f"normalizing (c2=1): {norm_c2[-1]}")
    print('----')

p1 = np.array(norm_c1)
p2 = np.array(norm_c2)
p1.shape, p2.shape = (4,1), (4,1)

w1 = np.sum(p1)
w2 = np.sum(p2)
print(f'\nw1 = {w1} ; w2 = {w2}\n')

mu1, mu2 = 0, 0
for i in range(4):
    mu1 = mu1 + 1/w1*(np.multiply(p1[i],x[i]))
    mu2 = mu2 + 1/w2*(np.multiply(p2[i],x[i]))
print(f'mu1 = {mu1}  ;   mu2 = {mu2}')

X_mat_c1, X_mat_c2 = [], []
for i in x:
    res_c1 = [[i[0]-mu1[0]], [i[1]-mu1[1]]]
    res_c2 = [[i[0]-mu2[0]], [i[1]-mu2[1]]]
    X_mat_c1.append(matmul(res_c1, transpose(res_c1)))
    X_mat_c2.append(matmul(res_c2, transpose(res_c2)))
X_mat_c1 = np.array(X_mat_c1)
X_mat_c2 = np.array(X_mat_c2)

for i in range(4):
    print(f'\n---- X{i+1}')
    print(f'Para c1=1: {X_mat_c1[i]}')
    print(f'Para c2=1: {X_mat_c2[i]}')

print('\n############\n')
Σ1, Σ2 = 0, 0
for i in range(len(p1)):
    Σ1 = np.add(Σ1, 1/w1*(np.multiply(p1[i], X_mat_c1[i])))
    Σ2 = np.add(Σ2, 1/w2*(np.multiply(p2[i], X_mat_c2[i])))
    
print(f'Σ1 = {Σ1}\n\nΣ2 = {Σ2}')

print(f'New Priors(W={w1+w2}): π1={w1/(w1+w2)}, π2={w2/(w1+w2)}')

print("\n###### VALS PROPRIOS ######\n")
eigvals1, eigvecs1 = la.eig(Σ1)
eigvals2, eigvecs2 = la.eig(Σ2)
print(f"eig1: {eigvals1}\n{eigvecs1}\n\neig2: {eigvals2}\n{eigvecs2}")

#Sketch clustering solution
x, y = np.mgrid[-5:7:0.01, -10:10:0.01]
pos = np.dstack((x,y))
mv_n1 = multivariate_normal(mu1, Σ1)
mv_n2 = multivariate_normal(mu2, Σ2)
plt.contour(x, y, mv_n1.pdf(pos), levels=8)
plt.contour(x, y, mv_n2.pdf(pos), levels=8)
plt.scatter(y1,y2, marker='o')
for i in range(len(y1)):
    plt.annotate(f'x{i+1}', (y1[i], y2[i]))
plt.show()

print("\n-------------- EX 2 ---------------\n")
data = [[2,4], [-1,-4], [-1,2], [4,0]]
def silhoutte_score(x, label):
    a, b = 0, 0
    for i in range(len(data)):
        if labels[i] == label:
            a += la.norm(np.subtract(x,data[i]))
        else:
            b += la.norm(np.subtract(x,data[i]))
    if (labels.count(label)!= 1):
        a = a/(labels.count(label)-1)
    if (len(labels)-labels.count(label) != 1):
        b = b/(len(labels)-labels.count(label)-1)
    return 1-(a/b)

labels = []
count_c1, count_c2 = 0, 0
for i in range(4):
    res1 = mv_n1.pdf((y1[i], y2[i]))
    res2 = mv_n2.pdf((y1[i], y2[i]))
    if res1 > res2:
        labels.append(1)
        count_c1 += 1
    else:
        labels.append(2)
        count_c2 += 1
print(f'Labels: {labels}')
silh_scores = []
for i in range(len(data)):
    silh_scores.append(silhoutte_score(data[i], labels[i]))
    print(f's(x{i+1}): {silhoutte_score(data[i], labels[i])}')

S_c1, S_c2 = 0, 0
for i in range(len(labels)):
    if labels[i] == 1:
        S_c1 += 1/(count_c1)*silh_scores[i]
    else:
        S_c2 += 1/(count_c2)*silh_scores[i]
print(f'\ns(c1): {S_c1}\ns(c2): {S_c2}')
print(f'\nsilhouette(C): {(S_c1+S_c2)/2}')