import numpy as np
import numpy.typing as npt

NDArray_float32 = npt.NDArray[np.float32]

# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w * x

# loss (MSE)
def loss(y: NDArray_float32, y_pred: NDArray_float32):
    return ((y_pred - y)**2).mean()

# gradient
# MSE = (1/N) * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
def gradient(x: NDArray_float32, y: NDArray_float32, y_pred: NDArray_float32):
    return np.dot(2*x, y_pred-y).mean()  # manually compute the gradient of the loss function. 

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction: forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")
