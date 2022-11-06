import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass
y_pred = w * x
loss = (y_pred - y)**2

print(loss)

# back prop
# pytorch automatically computes the local grads.
loss.backward()  # whole back prop
print(w.grad)

# updates weights
# next forward pass and back prop
# ...


