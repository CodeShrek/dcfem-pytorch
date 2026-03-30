import torch
import matplotlib.pyplot as plt

from problems.poisson_1d import solve_poisson


n = 50
D_true = 2.0

# Ground truth solution
with torch.no_grad():
    u_true = solve_poisson(n, D_true)

# Learnable parameter
D = torch.tensor(1.0, requires_grad=True)

optimizer = torch.optim.LBFGS([D], lr=1.0, max_iter=20)


# Loss Function (reusable))
def loss_fn(D_val):
    u_pred = solve_poisson(n, D_val)
    return ((u_pred - u_true)**2).mean()

#  Loop
def closure():
    optimizer.zero_grad()

    loss = loss_fn(D)
    loss.backward()

    return loss

for i in range(10):
    loss = optimizer.step(closure)
    print(f"Iter {i}: Loss={loss.item():.6f}, D={D.item():.4f}")


#Result
print("\nFinal Result:")
print(f"True D = {D_true}")
print(f"Learned D = {D.item():.6f}")

# Plots
u_pred = solve_poisson(n, D)

plt.plot(u_true.detach().numpy(), label="True")
plt.plot(u_pred.detach().numpy(), '--', label="Predicted")
plt.legend()
plt.title(f"Recovered D = {D.item():.4f}")
plt.savefig("result.png")   # saved in root directory
plt.show()



eps = 1e-4
D_val = D.item()

# Finite Difference Gradient
loss_plus = loss_fn(D_val + eps)
loss_minus = loss_fn(D_val - eps)

fd_grad = (loss_plus - loss_minus) / (2 * eps)

# Autograd Gradient 
D_check = torch.tensor(D_val, requires_grad=True)

loss = loss_fn(D_check)
loss.backward()

print("\nGradient Check:")
print(f"FD Grad: {fd_grad.item():.6e}")
print(f"Autograd Grad: {D_check.grad.item():.6e}")
