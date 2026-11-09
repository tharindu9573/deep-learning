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
lr = 0.2
input_neurons = 2
hidden_neurons = 6
beeta = 0.9
eps = 1e-8

he_std = np.sqrt(2.0 / (input_neurons))
W1 = np.random.randn(2, hidden_neurons) * he_std  # (2, 6)
b1 = np.zeros((1, hidden_neurons))
vW1, mW1 = np.zeros_like(W1), np.zeros_like(W1)
vb1, mb1 = np.zeros_like(b1), np.zeros_like(b1)

W2 = np.random.randn(hidden_neurons, 1)  # (6, 1)
b2 = np.zeros((1, 1))
vW2, mW2 = np.zeros_like(W2), np.zeros_like(W2)
vb2, mb2 = np.zeros_like(b2), np.zeros_like(b2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_derivated(x):
    return x > 0


for epoch in range(epoches):
    z1 = X.dot(W1) + b1

    a1 = relu(z1)  # (4, 6)

    z2 = a1.dot(W2) + b2

    a2 = sigmoid(z2)  # (4, 1)

    loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

    if loss < 0.001:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        break

    dz2 = (a2 - y) / 4  # (4, 1)

    dW2 = a1.T.dot(dz2)  # (6, 1)

    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2.dot(W2.T)  # (4, 6)

    dz1 = da1 * relu_derivated(a1)  # (4, 6)

    dW1 = X.T.dot(dz1)  # (2, 4)

    db1 = np.sum(dz1, axis=0, keepdims=True)

    vW2 = beeta*vW2 + (1 - beeta)*dW2**2
    vb2 = beeta*vb2 + (1 - beeta)*db2**2
    vW1 = beeta*vW1 + (1 - beeta)*dW1**2
    vb1 = beeta*vb1 + (1 - beeta)*db1**2

    W2 -= lr * dW2/(np.sqrt(vW2 + eps))
    b2 -= lr * db2/(np.sqrt(vb2 + eps))
    W1 -= lr * dW1/(np.sqrt(vW1 + eps))
    b1 -= lr * db1/(np.sqrt(vb1 + eps))

print("---------- Final prediction -----------")
print(a2.round(2))
