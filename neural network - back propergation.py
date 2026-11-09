# Solving the XOR Problem

#           |w1 w2|      a1            |w09|
# X1   =>   |w3 w4|(W1)  a2(a1)   =>   |w10|(w2)   =>  a2
# X2        |w5 w6|      a3            |w11|
#           |w7 w8|      a4            |w12|

# z1 = W1X + b1
# z2 = W2a1 + b2

import numpy as np

X = np.array([[0, 0],
              [1, 0],
              [1, 1],
              [0, 1]])  # (4, 2)

y = np.array([[0],
              [1],
              [0],
              [1]])  # (4,1)

np.random.seed(42)
epoches = 10000
lr = 0.5
hidden_layers = 6

W1 = np.random.randn(2, hidden_layers) * .5  # (2, 6)
b1 = np.zeros((1, hidden_layers))

W2 = np.random.randn(hidden_layers, 1)  # (6, 1)
b2 = np.zeros((1, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivated(s):
    return s * (1 - s)


for epoch in range(epoches):
    z1 = X.dot(W1) + b1

    a1 = sigmoid(z1)  # (4, 6)

    z2 = a1.dot(W2) + b2

    a2 = sigmoid(z2)  # (4, 1)

    loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

    dz2 = (a2 - y) / 4  # (4, 1)

    dW2 = a1.T.dot(dz2)  # (6, 1)

    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2.dot(W2.T)  # (4, 6)

    dz1 = da1 * sigmoid_derivated(a1)  # (4, 6)

    dW1 = X.T.dot(dz1)  # (2, 4)

    db1 = np.sum(dz1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("---------- Final prediction -----------")
print(a2.round(2))
