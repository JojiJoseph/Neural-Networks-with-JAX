# A program that fits a line to a set of data points
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from jax import grad, jit, vmap

def model(w, b, x):
    return w * x + b


x_train = jnp.arange(-1, 1, 0.1)
noise = np.random.normal(0, 0.1, x_train.shape)
m, c = 2 , 3

y_train = m * x_train + c + noise

plt.scatter(x_train, y_train)
plt.show()

# Non-jitted version
print("Non-jitted version")
w, b = 1, 1
lr = 0.001

def update(w, b, x):
    y_hat = model(w, b, x)
    loss = (y_hat - y_train) ** 2
    loss = jnp.mean(loss)
    w_grad, b_grad = 2 * (y_hat - y_train) * x_train, 2 * (y_hat - y_train)
    w_grad, b_grad = jnp.mean(w_grad), jnp.mean(b_grad)
    w -= lr * w_grad
    b -= lr * b_grad
    return w, b
start_time  = time.time()
for i in range(1000):
    w, b = update(w, b, x_train)
end_time = time.time()

print("Time taken: ", end_time - start_time)
x  = np.linspace(-1, 1, 100)
y = model(w, b, x)
plt.scatter(x_train, y_train)
plt.plot(x, y, color='red')
plt.show()

# Jitted version
print("Jitted version")
w, b = 1, 1
lr = 0.001

def update(w, b, x):
    y_hat = model(w, b, x)
    loss = (y_hat - y_train) ** 2
    loss = jnp.mean(loss)
    w_grad, b_grad = 2 * (y_hat - y_train) * x_train, 2 * (y_hat - y_train)
    w_grad, b_grad = jnp.mean(w_grad), jnp.mean(b_grad)
    w -= lr * w_grad
    b -= lr * b_grad
    return w, b
start_time  = time.time()
for i in range(1000):
    w, b = jit(update)(w, b, x_train)
end_time = time.time()

print("Time taken: ", end_time - start_time)
x  = np.linspace(-1, 1, 100)
y = model(w, b, x)
plt.scatter(x_train, y_train)
plt.plot(x, y, color='red')
plt.show()

# Grad version without jit
print("Grad version without jit")
def loss_fn(w, b, x):
    y_hat = model(w, b, x)
    loss = (y_hat - y_train) ** 2
    loss = jnp.mean(loss)
    return loss

loss_fn_grad = (grad(loss_fn, argnums=(0, 1)))

w, b = 1., 1.
lr = 0.001
start_time  = time.time()
for i in range(1000):
    w_grad, b_grad = loss_fn_grad(w, b, x_train)
    w -= lr * w_grad
    b -= lr * b_grad
end_time = time.time()
print("Time taken: ", end_time - start_time)
x  = np.linspace(-1, 1, 100)
y = model(w, b, x)
plt.scatter(x_train, y_train)
plt.plot(x, y, color='red')
plt.show()

# Grad version without jit
print("Grad version with jit")
def loss_fn(w, b, x):
    y_hat = model(w, b, x)
    loss = (y_hat - y_train) ** 2
    loss = jnp.mean(loss)
    return loss

loss_fn_grad = jit(grad(loss_fn, argnums=(0, 1)))

w, b = 1., 1.
lr = 0.001
start_time  = time.time()
for i in range(1000):
    w_grad, b_grad = loss_fn_grad(w, b, x_train)
    w -= lr * w_grad
    b -= lr * b_grad
end_time = time.time()
print("Time taken: ", end_time - start_time)
x  = np.linspace(-1, 1, 100)
y = model(w, b, x)
plt.scatter(x_train, y_train)
plt.plot(x, y, color='red')
plt.show()

