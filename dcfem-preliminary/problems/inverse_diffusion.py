import torch
from problems.poisson_1d import solve_poisson
n = 50

D_true = 2.0

with torch.no_grad():
    u_true = solve_poisson(n, D_true)

D = torch.tensor(1.0, requires_grad=True)

optimizer = torch.optim.LBFGS([D], lr=1.0, max_iter=20)

def closure():
    optimizer.zero_grad()

    u_pred = solve_poisson(n, D)

    loss = ((u_pred - u_true)**2).mean()
    loss.backward()

    return loss

for i in range(10):
    loss = optimizer.step(closure)
    print(f"Iter {i}: Loss={loss.item():.6f}, D={D.item():.4f}")

print("\nFinal Result:")
print(f"True D = {D_true}")
print(f"Learned D = {D.item():.6f}")

# Plot results
import matplotlib.pyplot as plt

u_pred = solve_poisson(n, D)

plt.plot(u_true.detach().numpy(), label="True")
plt.plot(u_pred.detach().numpy(), '--', label="Predicted")#note ->We detach tensors before visualization .
plt.legend()
plt.title(f"Recovered D = {D.item():.4f}")
plt.savefig("result.png")   # root directory 
plt.show()    