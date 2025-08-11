import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    params = [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    return params

def relu(x):
    return jnp.maximum(0, x)

def predict_single(params, x):
    """Predict single sample (regression output)"""
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    
    # Final layer (no activation for regression)
    final_w, final_b = params[-1]
    output = jnp.dot(final_w, activations) + final_b
    return output[0]  # Return scalar

# Vectorized prediction
predict_batch = vmap(predict_single, in_axes=(None, 0))

def mse_loss(params, x_batch, y_batch):
    predictions = predict_batch(params, x_batch)
    return jnp.mean((predictions - y_batch) ** 2)

@jit
def update_step(params, x_batch, y_batch, learning_rate):
    grads = grad(mse_loss)(params, x_batch, y_batch)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(params, grads)]

def square_root_baseline(volumes, signs):
    """Simple Square-Root Law baseline"""
    return 0.001 * jnp.sqrt(volumes) * signs

def propagator_baseline(volumes, signs, horizon=5):
    """Simple Propagator baseline""" 
    # G0(t) = Gamma0 * t0^beta / (t0 + t)^beta
    Gamma0, t0, beta = 0.001, 20, 0.38
    G0 = Gamma0 * (t0**beta) / ((t0 + horizon)**beta)
    return G0 * jnp.log(volumes + 1e-8) * signs

# Clean data
valid_idx = ~(features.isna().any(axis=1) | target.isna())
X = features[valid_idx].values
y = target[valid_idx].values

print(f"Valid samples: {len(X)}")
print(f"Features: {features.columns.tolist()}")

# Train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to JAX arrays
X_train = jnp.array(X_train.astype(np.float32))
X_test = jnp.array(X_test.astype(np.float32))
y_train = jnp.array(y_train.astype(np.float32))
y_test = jnp.array(y_test.astype(np.float32))

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Network architecture
layer_sizes = [X_train.shape[1], 16, 8, 1]  # input_dim -> 16 -> 8 -> 1
learning_rate = 0.001
n_epochs = 500
batch_size = 256

print(f"Architecture: {layer_sizes}")

# Initialize network
key = random.PRNGKey(42)
params = init_network_params(layer_sizes, key)

# Training loop
n_batches = len(X_train) // batch_size

for epoch in range(n_epochs):
    # Mini-batch training
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        params = update_step(params, X_batch, y_batch, learning_rate)
    
    # Print progress
    if epoch % 100 == 0:
        train_loss = mse_loss(params, X_train, y_train)
        test_loss = mse_loss(params, X_test, y_test)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

# Get predictions
nn_pred = predict_batch(params, X_test)

# Traditional baselines (using existing functions)
test_volumes = jnp.exp(X_test[:, 0])  # Convert back from log_volume
test_signs = X_test[:, 1]  # trade_sign

sqrt_pred = square_root_baseline(test_volumes, test_signs)
prop_pred = propagator_baseline(test_volumes, test_signs)

# Calculate metrics
def calc_r2(y_true, y_pred):
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

print("\nMODEL PERFORMANCE:")
print("-" * 30)

models = {
    'Neural Network': nn_pred,
    'Square Root Law': sqrt_pred[:len(y_test)],
    'Propagator Model': prop_pred[:len(y_test)]
}

for name, pred in models.items():
    mse = jnp.mean((y_test - pred) ** 2)
    r2 = calc_r2(y_test, pred)
    print(f"{name:15}: MSE = {mse:.2e}, R² = {r2:.4f}")
