import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time
from flax import linen as nn
import matplotlib.pyplot as plt


# Define the model
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input shape: (batch_size, 28 * 28, 1)
        x = nn.Conv(features=32, kernel_size=(5, 5), padding="VALID")(
            x
        )  # (batch_size, 24 * 24, 32)
        x = nn.relu(x)
        x = nn.max_pool(
            x, window_shape=(2, 2), strides=(2, 2)
        )  # (batch_size, 12 * 12, 32)
        x = nn.Conv(features=64, kernel_size=(5, 5), padding="VALID")(
            x
        )  # (batch_size, 8 * 8, 64)
        x = nn.relu(x)
        x = nn.max_pool(
            x, window_shape=(2, 2), strides=(2, 2)
        )  # (batch_size, 4 * 4, 64)
        x = x.reshape((x.shape[0], -1))  # (batch_size, 4 * 4 * 64 = 1024)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.softmax(x)
        return x


model = CNN()
print(model.tabulate(jax.random.PRNGKey(1), x=jnp.ones((1, 28, 28, 1))))

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
y = np.array(list(map(int, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X_train = X_train.astype(np.float32).reshape((-1, 28, 28, 1)) / 255.0
X_test = X_test.astype(np.float32).reshape((-1, 28, 28, 1)) / 255.0

params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))["params"]
out = model.apply({"params": params}, jnp.ones((1, 28, 28, 1)))


def categorical_cross_entropy(y, y_hat, n_classes=10, one_hot=True):
    if one_hot:
        return -jnp.sum(y * jnp.log(y_hat))

    else:
        y = jax.nn.one_hot(y, n_classes)
        return -jnp.sum(y * jnp.log(y_hat))


def loss_fn(params, x, y):
    y_hat = model.apply({"params": params}, x)
    print(y_hat.shape)
    return categorical_cross_entropy(y, y_hat, one_hot=False) / float(y.shape[0])


def update(params, x, y, lr):
    grads = jax.grad(loss_fn, argnums=(0,))(params, x, y)
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads[0])
    return params


def accuracy(params, x, y):
    y_hat = model.apply({"params": params}, x)
    return jnp.mean(jnp.argmax(y_hat, axis=1) == y)


# Train the model
start_time = time.time()
for epoch in range(20):
    for i in range(0, len(X_train), 64):
        X_batch = X_train[i : i + 64]
        y_batch = y_train[i : i + 64]
        params = jax.jit(update)(params, X_batch, y_batch, 1e-3)
    print(f"Epoch {epoch}, test accuracy: {accuracy(params, X_test, y_test)}")
end_time = time.time()
print("Time taken: ", end_time - start_time, " seconds")

# Test if the inference is working

X_batch = X_test[:10]
y_batch = y_test[:10]


plt.subplots(2, 5, figsize=(10, 5))
for i, (X_img, y_img) in enumerate(zip(X_batch, y_batch)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_img.reshape(28, 28), cmap="gray")
    plt.title(
        f"Label: {y_img} \n Prediction: {jnp.argmax(model.apply({'params':params}, X_img[None]))}"
    )
plt.show()
