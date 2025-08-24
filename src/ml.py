import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap, random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from data_processing import feature_columns
from bouchaud_analysis import propagator_function

#==================================================================================
# Utility methods
#==================================================================================

def init_layer_params(m, n, key, scale=1e-2):
    """Initialize parameters for a single layer."""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network(layer_sizes, key):
    """Initialize neural network parameters."""
    keys = random.split(key, len(layer_sizes))
    return [init_layer_params(m, n, k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]

def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)

def predict_single(params, x):
    """Forward pass for a single sample."""
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    
    # Final layer (no activation for regression)
    final_w, final_b = params[-1]
    output = jnp.dot(final_w, activations) + final_b
    return output[0]

# Vectorized prediction
predict_batch = vmap(predict_single, in_axes=(None, 0))

def mse_loss(params, x_batch, y_batch):
    """Mean squared error loss function."""
    predictions = predict_batch(params, x_batch)
    return jnp.mean((predictions - y_batch) ** 2)

@jit
def update_params(params, x_batch, y_batch, learning_rate):
    """Single gradient update step."""
    grads = grad(mse_loss)(params, x_batch, y_batch)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(params, grads)]

def train(X, y, layer_sizes=None, learning_rate=0.001, n_epochs=500, 
                        batch_size=256, random_seed=42):
    """
    Train a neural network with given data.
    
    Args:
        X: Feature matrix
        y: Target vector
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        learning_rate: Learning rate for optimization
        n_epochs: Number of training epochs
        batch_size: Batch size for mini-batch training
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (trained_params, scaler) where params are network weights and scaler is fitted StandardScaler
    """
    # Set default architecture
    if layer_sizes is None:
        layer_sizes = [X.shape[1], 16, 8, 1]
    else:
        layer_sizes[0] = X.shape[1]  # Ensure input dimension matches data
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to JAX arrays
    X_train = jnp.array(X_scaled.astype(np.float32))
    y_train = jnp.array(y.astype(np.float32))
    
    # Initialize network
    key = random.PRNGKey(random_seed)
    params = init_network(layer_sizes, key)
    
    # Training loop
    n_batches = len(X_train) // batch_size
    
    for epoch in range(n_epochs):
        # Mini-batch training
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            params = update_params(params, X_batch, y_batch, learning_rate)
        
        if epoch % 100 == 0:
            train_loss = mse_loss(params, X_train, y_train)
            train_acc = jnp.mean((predict_batch(params, X_train) > 0.5) == y_train)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.6f}")

    return params, scaler

def predict_neural_network(params, scaler, X):
    """Make predictions with trained neural network."""
    X_scaled = scaler.transform(X)
    X_test = jnp.array(X_scaled.astype(np.float32))
    return predict_batch(params, X_test)

#==================================================================================
# Baseline Models
#==================================================================================

def square_root_baseline(volumes, signs):
    """Square-Root Law baseline model."""
    return 0.001 * jnp.sqrt(volumes) * signs

def propagator_baseline(volumes, signs, horizon=5, t0=20, beta=0.38, Gamma0=0.001):
    """Propagator model baseline."""
    G0 = propagator_function(horizon, t0, beta, Gamma0)
    return G0 * jnp.log(volumes + 1e-8) * signs

#==================================================================================
# Model Evaluation
#==================================================================================

def calc_r2(y_true, y_pred):
    """Calculate R-squared score."""
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def calc_mse(y_true, y_pred):
    """Calculate mean squared error."""
    return jnp.mean((y_true - y_pred) ** 2)

def evaluate_all_models(X_test, y_test, nn_params=None, nn_scaler=None):
    """
    Evaluate all models and return performance metrics.
    
    Args:
        X_test: Test features
        y_test: Test targets
        nn_params: Trained neural network parameters (optional)
        nn_scaler: Fitted scaler for neural network (optional)
    
    Returns:
        dict: Performance metrics for all models
    """
    results = {}
    
    # Neural Network predictions
    if nn_params is not None and nn_scaler is not None:
        nn_pred = predict_neural_network(nn_params, nn_scaler, X_test)
        results['Neural Network'] = {
            'predictions': nn_pred,
            'accuracy': float(jnp.mean((nn_pred > 0.5) == y_test)),
            'mse': float(calc_mse(y_test, nn_pred)),
            'r2': float(calc_r2(y_test, nn_pred))
        }
    
    # Baseline models (assuming first two features are log_volume and trade_sign)
    test_volumes = jnp.exp(X_test[:, 0])  # Convert back from log_volume
    test_signs = X_test[:, 1]  # trade_sign
    
    sqrt_pred = square_root_baseline(test_volumes, test_signs)
    prop_pred = propagator_baseline(test_volumes, test_signs)
    
    results['Square Root Law'] = {
        'predictions': sqrt_pred,
        'accuracy': float(jnp.mean((sqrt_pred > 0.5) == y_test)),
        'mse': float(calc_mse(y_test, sqrt_pred)),
        'r2': float(calc_r2(y_test, sqrt_pred))
    }
    
    results['Propagator Model'] = {
        'predictions': prop_pred,
        'accuracy': float(jnp.mean((prop_pred > 0.5) == y_test)),
        'mse': float(calc_mse(y_test, prop_pred)),
        'r2': float(calc_r2(y_test, prop_pred))
    }
    
    return results

def print_model_performance(results):
    """Print model performance comparison."""
    print("\nMODEL PERFORMANCE:")
    print("-" * 50)
    
    for name, metrics in results.items():
        mse = metrics['mse']
        r2 = metrics['r2']
        print(f"{name:20}: MSE = {mse:.2e}, R² = {r2:.4f}")

#==================================================================================
# Data Processing Functions
#==================================================================================

def prep_data(data_file='../data/features_test.csv', target_col='price_change_5', test_split=0.8):

    # Load data
    df = pd.read_csv(data_file)
    
    # Get feature columns
    feature_cols = feature_columns(df, exclude_targets=True)
    
    # Prepare features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Train/test split
    split_idx = int(len(X) * test_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols

#==================================================================================

if __name__ == "__main__":

    layer_sizes = [14*14, 10, 10, 10]
    learning_rate = 0.001
    step_size = 0.01
    batch_epochs = 100
    num_epochs = batch_epochs * 10
    batch_size = 128
    params = init_network(layer_sizes, random.PRNGKey(0))

    # Load and prepare data
    print("\n1. Loading data...")
    X_train, X_test, y_train, y_test, feature_cols  = prep_data('../data/features_test.csv', 'price_change_5')
    
    print(f"Valid samples: {len(X_train) + len(X_test)}")
    print(f"Features: {feature_cols}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train neural network
    print("\n2. Training...")
    nn_params, nn_scaler = train(
        X_train, y_train, layer_sizes, learning_rate, num_epochs, batch_size
    )
    
    # Evaluate all models
    print("\n3. Evaluating models...")
    results = evaluate_all_models(X_test, y_test, nn_params, nn_scaler)
    print_model_performance(results)

    results = {
        'nn_params': nn_params,
        'nn_scaler': nn_scaler,
        'results': results,
        'feature_cols': feature_cols,
        'test_data': (X_test, y_test),
        'train_data': (X_train, y_train)
    }