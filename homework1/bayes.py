import numpy
from statistics import mean
from statistics import variance
from statistics import stdev
from fractions import Fraction
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

y1 = [0.6, 0.1, 0.2, 0.1, 0.3, -0.1, -0.3, 0.2, 0.4, -0.2]
y2 = ['A' ,'B' ,'A' ,'C' ,'B' ,'C' ,'C' ,'B' ,'A' ,'C']
y3 = [0.2, -0.1, -0.1, 0.8, 0.1, 0.2, -0.1, 0.5, -0.4, 0.4]
y4 = [0.4, -0.4, 0.2 , 0.8, 0.3, -0.2, 0.2, 0.6, -0.7, 0.3]

y1_if_c0, y2_if_c0, y3_if_c0, y4_if_c0 = y1[:4], y2[:4], y3[:4], y4[:4]
y1_if_c1, y2_if_c1, y3_if_c1, y4_if_c1 = y1[4:], y2[4:], y3[4:], y4[4:]
p_c0, p_c1 = 4/10, 6/10

actual,pred = [0,0,0,0,1,1,1,1,1,1], []

p_y1_c0, p_y1_c1, p_y2_c0, p_y2_c1, p_y3y4_c0, p_y3y4_c1 = ([] for _ in range(6))

def p_y2_if_c0(x):
    return float(y2_if_c0.count(x))/len(y2_if_c0)


def p_y2_if_c1(x):
    return float(y2_if_c1.count(x))/len(y2_if_c1)

print('\n\n #####  ALINEA A)  ##### \n')
print("---------- c = 0 ----------\n")
y1_if_c0_mean = mean(y1_if_c0)
y1_if_c0_var = variance(y1_if_c0)
y1_if_c0_dev = stdev(y1_if_c0)

print('- y1 | c = 0 ~ N(µ = {}, σ^2 = {})\n'.format(y1_if_c0_mean, y1_if_c0_var))
print("- P(y2 = A | c = 0) = " + str(Fraction(p_y2_if_c0("A"))))
print("P(y2 = B | c = 0) = " + str(Fraction(p_y2_if_c0("B"))))
print("P(y2 = C | c = 0) = " + str(Fraction(p_y2_if_c0("C"))) + '\n')

y3_if_c0_mean = mean(y3_if_c0)
y4_if_c0_mean = mean(y4_if_c0)

print('- y3,y4 | c = 0 ~ N(µ = {}, Σ = {})\n'.format([y3_if_c0_mean, y4_if_c0_mean], numpy.cov(y3_if_c0,y4_if_c0)))

print("---------- c = 1 ----------\n")
y1_if_c1_mean = mean(y1_if_c1)
y1_if_c1_var = variance(y1_if_c1)
y1_if_c1_dev = stdev(y1_if_c1)

print('- y1 | c = 1 ~ N(µ = {}, σ^2 = {})\n'.format(y1_if_c1_mean, y1_if_c1_var))

print("- P(y2 = A | c = 1) = " + str(Fraction(p_y2_if_c1("A")).limit_denominator()))
print("P(y2 = B | c = 1) = " + str(Fraction(p_y2_if_c1("B")).limit_denominator()))
print("P(y2 = C | c = 1) = " + str(Fraction(p_y2_if_c1("C"))) + '\n')

y3_if_c1_mean = mean(y3_if_c1)
y4_if_c1_mean = mean(y4_if_c1)

print('- y3,y4 | c = 1 ~ N(µ = {}, Σ = {})\n'.format([y3_if_c1_mean, y4_if_c1_mean], numpy.cov(y3_if_c1,y4_if_c1)))

print('\n\n #####  ALINEA B)  ##### \n')
for i in range(0,10):
    p_y1_c0.append(norm(y1_if_c0_mean, y1_if_c0_dev).pdf(y1[i]))
    p_y2_c0.append(p_y2_if_c0(y2[i]))
    p_y3y4_c0.append(multivariate_normal(mean=[y3_if_c0_mean, y4_if_c0_mean], cov = numpy.cov(y3_if_c0, y4_if_c0)).pdf([y3[i], y4[i]]))

    p_y1_c1.append(norm(y1_if_c1_mean, y1_if_c1_dev).pdf(y1[i]))
    p_y2_c1.append(p_y2_if_c1(y2[i]))
    p_y3y4_c1.append(multivariate_normal(mean = [y3_if_c1_mean, y4_if_c1_mean], cov = numpy.cov(y3_if_c1, y4_if_c1)).pdf([y3[i], y4[i]]))
    
    print("P(c = 0 | " + "x" + str(i+1) + ")= " + str(p_y1_c0[i]*p_y2_c0[i]*p_y3y4_c0[i]*p_c0))
    print("P(c = 1 | " + "x" + str(i+1) + ")= " + str(p_y1_c1[i]*p_y2_c1[i]*p_y3y4_c1[i]*p_c1))
    
    if p_y1_c0[i]*p_y2_c0[i]*p_y3y4_c0[i]*p_c0 > p_y1_c1[i]*p_y2_c1[i]*p_y3y4_c1[i]*p_c1:
        pred.append(0)
        print("class = 0\n")

    else:
        pred.append(1)
        print("class = 1\n")
    

print('\n\n #####  ALINEA C)  ##### \n')
print("Actual: {}\t Pred: {}".format(actual, pred))
fp, fn, tp, tn = 0, 0, 0, 0
for i in range(10):
    if actual[i] == pred[i]:
        if (actual[i] == 0):
            tn += 1
        else:
            tp +=1
    else:
        if actual[i] == 0 and pred[i] == 1:
            fp += 1
        else: # if actual[i] == 1 & pred[i] == 0
            fn += 1

print("FP: {}, FN: {}, TP: {} TN: {}\n".format(fp,fn,tp,tn))

print("F1 = {}".format(f1_score(actual, pred)))

print('\n\n #####  ALINEA D)  ##### \n')
for i in range(0, 10):
    p_xnew_c0 = p_y1_c0[i]*p_y2_c0[i]*p_y3y4_c0[i]
    p_xnew_c1 = p_y1_c1[i]*p_y2_c1[i]*p_y3y4_c1[i]

    print("P(class = 1 | x{}) = {}".format(i+1, p_xnew_c1*p_c1/(p_xnew_c1*p_c1 + p_xnew_c0*p_c0)))
