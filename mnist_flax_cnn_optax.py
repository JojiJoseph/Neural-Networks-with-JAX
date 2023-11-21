import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import time
from flax import linen as nn
import matplotlib.pyplot as plt

from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
import optax  # Common loss functions and optimizers


# TODO - make use of the following two classes later
@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, 28, 28, 1]))[
        "params"
    ]  # initialize parameters by passing a template image
    tx = optax.adam(learning_rate)  # , momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


init_rng = jax.random.PRNGKey(0)

learning_rate = 0.01
momentum = 0.9


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
        # x = nn.softmax(x)
        return x


model = CNN()
print(model.tabulate(jax.random.PRNGKey(1), x=jnp.ones((1, 28, 28, 1))))
state = create_train_state(model, init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
y = np.array(list(map(int, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X_train = X_train.astype(np.float32).reshape((-1, 28, 28, 1)) / 255.0
X_test = X_test.astype(np.float32).reshape((-1, 28, 28, 1)) / 255.0

params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))["params"]
out = model.apply({"params": params}, jnp.ones((1, 28, 28, 1)))


def accuracy(params, x, y):
    y_hat = model.apply({"params": params}, x)
    return jnp.mean(jnp.argmax(y_hat, axis=1) == y)


# Train the model
start_time = time.time()
for epoch in range(20):
    for i in range(0, len(X_train), 64):
        # X_batch = X_train[i : i + 64]
        # y_batch = y_train[i : i + 64]
        batch = {"image": X_train[i : i + 64], "label": y_train[i : i + 64]}
        state = train_step(state, batch)
        print(state.params['Conv_0'].keys())
        # params = jax.jit(update)(params, X_batch, y_batch, 1e-3)
    print(f"Epoch {epoch}, test accuracy: {accuracy(state.params, X_test, y_test)}")
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
        f"Label: {y_img} \n Prediction: {jnp.argmax(model.apply({'params':state.params}, X_img[None]))}"
    )
plt.show()
