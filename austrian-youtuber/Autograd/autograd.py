import torch

# autograd calculates gradients
# pytorch provides auto grad package. 

x = torch.randn(3, requires_grad=True)
print(x)

# whenever we do operations with this tensor, pytorch will create a computational graph to compute the gradient
y = x + 2  # creates the computational graph.
# this allows to calculate gradients on back propagation. 
# pytorch will automatically create a derivate function for each operation on the operation node.
# Forward Pass: pytorch just computes and, creates and stores the derivative function 
# Back Prop: dy/dx. For, node of x 2 +
print(y)
z = y * y * 2
print(z)
z = z.mean()
print(z)

z.backward()  # dz/dx
print(x.grad)
# print(y.grad) # not a leaf in the computational tree

# pytorch uses a vector jacobian product to get the gradients. This is in essence the chain rule. 

# here z was a scalar. Gradients can be computer for scalar outputs. 

# if output is not a scalar, we have to provide the gradient argument (a vector of the same size)
#  to the .backward() call. 
z1 = y * y * 2  # 3x1
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z1.backward(v)  # gradient arg
# in the background this is a vector jacobian. 

# most cases the last operation creates a scalar value. But if that is not a scalar, we must give it a vector. 

# we can stop the gradient function creation, 
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad(): 


# whenver we call the .backward() function - the gradient for this tensor will be accumulated on the 
# .grad attribute. 

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)  # for multiple iterations, this will be the accumulation of all the grads. Will be incorrect!

    # so before we do the next optimization step, we must empty the gradient. 
    if weights.grad:
        weights.grad.zero_()


