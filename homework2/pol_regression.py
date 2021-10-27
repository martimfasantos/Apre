from numpy.linalg import norm 
from numpy import transpose, matmul
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error

# Learning Data
y0 = [1,1,1,1,1,1,1,1]
y1 = [1,1,0,1,2,1,2,0]
y2 = [1,1,2,2,0,1,0,2]
y3 = [0,5,4,3,7,1,2,9]
output = [1,3,2,0,6,4,5,7]

# Testing Data
y0_t = [1,1]
y1_t = [2,1]
y2_t = [0,2]
y3_t = [0,1]
output_t = [2,4]

X_mat = []

# Create X vector for learning
x_l = []
for i in range(8):
    x_l.append([y1[i], y2[i], y3[i]])

# Create X vector for testing
x_t = []
for i in range(2):
    x_t.append([y1_t[i], y2_t[i], y3_t[i]])

# Basis Function
def phi_func(x):
    return norm(x,2)

# Create line with phi_fun
def create_line(x):
    return [1, round(phi_func(x), 5), round(phi_func(x)**2, 5), round(phi_func(x)**3, 5)]

# w = (X^T·X)^-1 · X^T · Z
def calculate_w(X, Z):
    return inv(transpose(X).dot(X)).dot(transpose(X)).dot(Z)

# info and adding lines to matrix X (X_mat)
index = 0
for i in x_l:
    line = create_line(i)
    X_mat.append(line)
    print("f(x{}, w) = w0 + w1*{} + w2*{} + w3*{}".format(index+1, line[1], line[2], line[3]))  
    index += 1

print("\n##############\n")

# print w vector and A
w = calculate_w(X_mat, transpose(output))
print("w = {}".format(w))
print("\n##############\n")
x9, x10 = create_line(x_t[0]), create_line(x_t[1])
pred_output = matmul([x9, x10], transpose(w)) # [pred_x9, pred_x10]
print("[pred_x9, pred_x10] = {}".format(pred_output))
print("RMSE = {}".format(mean_squared_error(output_t, pred_output, squared=False)))