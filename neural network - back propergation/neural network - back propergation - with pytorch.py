# Solving the XOR Problem

#           |w1 w2|      a1            |w09|
# X1   =>   |w3 w4|(W1)  a2(a1)   =>   |w10|(w2)   =>  a2
# X2        |w5 w6|      a3            |w11|
#           |w7 w8|      a4            |w12|

# z1 = W1X + b1
# z2 = W2a1 + b2


import torch
import torch.nn as nn

X = torch.tensor([[0., 0.],
                  [1., 0.],
                  [1., 1.],
                  [0., 1.]])  # (4, 2)

y = torch.tensor([[0.],
                  [1.],
                  [0.],
                  [1.]])  # (4,1)


model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(4000):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimiser.zero_grad()
    loss.backward()

    optimiser.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("\nFinal prediction:")
print(model(X).round())
