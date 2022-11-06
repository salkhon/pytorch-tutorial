import torch
import numpy as np

x = torch.empty(3, 2, 4, dtype=torch.int)
print(x)
print(x.dtype)
print(x.size())

y = torch.tensor([2.5, 0.1])
print(y)

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)

z = x + y
print(z)
z = torch.add(x, y)
print(z)

# in pytorch, every function with trailing _, will do an inplace operation. 
# add_ means += 
y.add_(x)
print(y)

# slicing
x = torch.rand(5, 3)
print(x)
print(x[:, 0])

# accessing element
print(x[2, 2])  # returns tensor element
print(x[2, 2].item())  # returns actual element

# reshape
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)

# numpy to torch
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
# *** if the tensor is on the cpu and not the gpu, the tensor and ndarray will share the same memory location. 

a.add_(1)
print(a)
print(b)
# changes both

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
a += 1
print(a)
print(b)

# same memory unless on the GPU. 
# needs CUDA tool
print(torch.cuda.is_available())

# *** cuda routine
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y  # will be performed in the GPU, and is much master
    # but careful, 
    # z.numpy() will return an error. NUMPY can only CPU tensors. 
    z = z.to("cpu")  # move back to CPU
    
# tells pytorch that you will need to calculate the gradient of this tensors in your optimization steps. 
x = torch.ones(5, requires_grad=True)
print(x)    

    