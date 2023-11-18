import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time


# Layers
def conv2d(W, x):
    # x is of shape (height, width, channels)
    return jax.lax.conv_general_dilated(x[None], W, window_strides=(1, 1), padding='VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
def linear(W, b, x):
    return jnp.dot(W, x) + b

def relu(x):
    return jnp.maximum(0, x)

def softmax(x):
    return jnp.exp(x) / (jnp.sum(jnp.exp(x)) + 1e-9)

def categorical_cross_entropy(y, y_hat, n_classes=10, one_hot=True):
    if one_hot:
        return -jnp.sum(y * jnp.log(y_hat))
        
    else:
        y = jax.nn.one_hot(y, n_classes) 
        return -jnp.sum(y * jnp.log(y_hat))

# Define the model
def model(params, x):
    x = conv2d(params['C1'], x)[0]
    x = relu(x)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (2, 2, 1), (2, 2, 1), padding='VALID')
    x = conv2d(params['C2'], x)[0]
    x = relu(x)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (2, 2, 1), (2, 2, 1), padding='VALID')

    x = x.reshape((-1, ))
    x = linear(params['W1'], params['b1'], x)
    x = relu(x)
    x = linear(params['W2'], params['b2'], x)
    x = relu(x)
    x = linear(params['W3'], params['b3'], x)
    x = softmax(x)
    return x

# Define the loss function
def loss_fn(params, x, y):
    y_hat = vmap(model, in_axes=[None, 0])(params, x)
    return categorical_cross_entropy(y, y_hat, one_hot=False) / float(y.shape[0])

# Define the update function
def update(params, x, y, lr):
    grads = jax.grad(loss_fn, argnums=(0, ))(params, x, y)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads[0])
    return params

X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)
y = np.array(list(map(int, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X_train = X_train.astype(np.float32).reshape((-1,28, 28, 1)) / 255.
X_test = X_test.astype(np.float32).reshape((-1,28, 28, 1)) / 255.


# Create params 
params = {
    "C1": np.random.normal(0., 0.1, (5, 5, 1, 32)),
    "C2": np.random.normal(0., 0.1, (5, 5, 32, 64)),
    "W1": np.random.normal(0., 0.1, (128, 1024)),
    "b1": np.random.normal(0., 0.1, (128, )),
    "W2": np.random.normal(0., 0.1, (128, 128)),
    "b2": np.random.normal(0., 0.1, (128, )),
    "W3": np.random.normal(0., 0.1, (10, 128)),
    "b3": np.random.normal(0., 0.1, (10, )),
}


def accuracy(params, x, y):
    y_hat = vmap(model, in_axes=[None, 0])(params, x)
    return jnp.mean(jnp.argmax(y_hat, axis=1) == y)



# Train the model
start_time = time.time()
for epoch in range(20):
    for i in range(0, len(X_train), 64):
        X_batch = X_train[i:i+64]
        y_batch = y_train[i:i+64]
        params = jax.jit(update)(params, X_batch, y_batch, 1e-3)
    print(f"Epoch {epoch}, test accuracy: {accuracy(params, X_test, y_test)}")
end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")

# Test if the inference is working

X_batch = X_test[:10]
y_batch = y_test[:10]

import matplotlib.pyplot as plt
plt.subplots(2, 5, figsize=(10, 5))
for i, (X_img, y_img) in enumerate(zip(X_batch, y_batch)):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_img.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_img} \n Prediction: {jnp.argmax(model(params, X_img))}")
plt.show()