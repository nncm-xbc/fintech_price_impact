<div align="center">

# **Fintech Price Impact Analysis**

[Installation](#installation) • [Usage](#usage)

</div>
This project implements a comprehensive analysis of market impact dynamics, progressing from traditional static models to advanced temporal effects and machine learning approaches. The work is based on the propagator model framework introduced by J.P. Bouchaud in "Fluctuations and response in financial markets: The subtle nature of 'random' price changes."

## Project Overview

The project analyzes market impact through four main phases:

1. **Baseline Implementation**: Square-Root Law (ΔP = Y σ√(Q/V)) with statistical validation
2. **Traditional Model Extensions**: Propagator Model (P(t) = ∑G(t-s)ε(s)) and temporal impact analysis
3. **Machine Learning Implementation**: Neural network prediction of price impact using microstructure features
4. **Regime-Dependent Analysis**: Cross-validation across different market conditions

## Data Sources and Processing

### Data Source

The project uses high-frequency cryptocurrency trading data from Binance vision:
https://data.binance.vision/?prefix=data/spot/daily/trades/BTCUSDT/
https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1s/

- **Trade Data**: Individual BTCUSDT trades with nanosecond timestamps
  - Files: `BTCUSDT-trades*.csv` in `data/binance_raw/`
  - Contains: trade_id, price, volume, quote_volume, timestamp, is_buyer_maker, is_best_match

- **Quote Data**: 1-second OHLCV candle data
  - Files: `BTCUSDT-1s*.csv` in `data/binance_raw/`
  - Contains: open_time, open, high, low, close, volume, quote_volume, count

### Data Processing Pipeline

The data processing follows these steps:

1. **Collection and Consolidation**
   - Concatenates multiple raw CSV files into unified datasets
   - Converts timestamps to datetime format
   - Derives trade signs from `is_buyer_maker` field (-1 for seller-initiated, +1 for buyer-initiated)

2. **Volume Normalization**
   - Log-normalizes volume to handle extreme values
   - Creates `log_volume` and `sqrt_volume` features
   - Scales by median volume to avoid numerical issues

3. **Quote Matching**
   - Merges trade data with nearest quote data using `merge_asof`
   - Calculates midpoint prices and spreads
   - Preserves temporal relationships between trades and market state

4. **Sparsification** (Optional)
   - Groups trades by timestamp and sign
   - Volume-weighted price averaging for simultaneous trades
   - Reduces data density while preserving market impact signal

### Feature Engineering

The feature extraction creates 30+ microstructure features across multiple time windows (5, 20, 50 trades):

**Core Market Features:**
- Price returns and log returns
- Volatility metrics (rolling standard deviation)
- Volume patterns and distributions
- Trade sign patterns and autocorrelations

**Market Microstructure Features:**
- Bid-ask spreads and midpoint deviations
- Order flow imbalance metrics
- Trade frequency and intensity measures
- Price range and high-low volatility

**Advanced Features:**
- Response function approximations
- Sign correlation decay (gamma parameter)
- Effective trade measures
- Local diffusion vs response strength
- Critical behavior indicators
- Market regime indicators (price trends)

## Analysis Modules

### Bouchaud Analysis (`src/bouchaud_analysis.py`)

Implements the theoretical framework from Bouchaud's paper:

**Core Functions:**
- **Response Function R(ℓ)**: Measures average price impact at different time lags
- **Sign Correlations C₀(ℓ)**: Analyzes persistence of trade sign patterns
- **Volume-Conditioned Response**: Impact analysis segmented by trade size
- **Power Law Fitting**: Extracts decay exponents from correlation functions
- **Critical Behavior Analysis**: Tests theoretical predictions around β_c = (1-γ)/2

**Outputs:**
- Response function plots with theoretical fits
- Correlation function analysis with power law decay
- Critical exponent estimation and validation
- Volume conditioning effects visualization

### Machine Learning Module (`src/ml.py`)

Neural network implementation using JAX for high-performance computation:

**Architecture:**
- Fully connected layers with ReLU activation
- Robust feature scaling using RobustScaler
- JAX-based automatic differentiation and JIT compilation
- Configurable architecture (default: input → 64 → 32 → 16 → 1)

**Training:**
- Mini-batch training with learning rate decay
- Mean squared error loss function
- Temporal train/test split to avoid look-ahead bias
- Model persistence with pickle serialization

**Baselines:**
- Square-Root Law: Simple ΔP = σ√V × sign model
- Propagator Model: Uses theoretical propagator function with learned parameters
- Performance comparison using MSE and R² metrics

## Installation and Dependencies

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster JAX computation)

### Required Packages

```bash
# Core scientific computing
pip install numpy pandas scipy matplotlib

# Machine learning and statistics
pip install scikit-learn jax jaxlib

# Data processing utilities
pip install joblib

# For GPU support (optional)
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Alternative Installation

```bash
# Create virtual environment
python -m venv market_impact_env
source market_impact_env/bin/activate  # On Windows: market_impact_env\Scripts\activate

# Install all dependencies
pip install numpy pandas scipy matplotlib scikit-learn jax jaxlib joblib
```

### Project Structure

```
project_root/
├── README.md
├── src/
│   ├── data_processing.py      # Data collection and feature engineering
│   ├── bouchaud_analysis.py    # Theoretical analysis implementation
│   └── ml.py                   # Machine learning models and evaluation
├── data/
│   ├── binance_raw/           # Raw Binance data files
│   ├── trades_dense.csv       # Processed trade data
│   ├── trades_sparse.csv      # Sparsified trade data
│   ├── ohlvc.csv             # Quote data
│   ├── trades_w_quotes.csv   # Merged trades and quotes
│   ├── features_test.csv     # Engineered features
│   └── target_test.csv       # Target variables
├── models/
│   ├── nn_params.pkl         # Trained neural network parameters
│   └── nn_scaler.pkl         # Feature scaling parameters
└── plots/                    # Generated analysis plots
```

## Usage Instructions

### 1. Data Processing

Process raw Binance data into features and targets:

```bash
cd src/
python data_processing.py
```

**Options:**
- `--no-collect`: Skip data collection step if processed files already exist

**Outputs:**
- `../data/features_test.csv`: Engineered features matrix
- `../data/target_test.csv`: Price impact targets
- `../data/trades_w_quotes.csv`: Merged trade and quote data

### 2. Theoretical Analysis

Run Bouchaud's market microstructure analysis:

```bash
cd src/
python bouchaud_analysis.py
```

**Outputs:**
- Response function analysis with theoretical fits
- Sign correlation analysis with power law extraction
- Critical behavior validation
- Plots saved to `../plots/` directory:
  - `response_function.png`
  - `correlation_functions.png`
  - `conditioned_response_function.png`
  - `fit_quality.png`

### 3. Machine Learning Training and Evaluation

Train neural network and compare with baseline models:

```bash
cd src/
python ml.py
```

**Training Process:**
1. Loads feature and target data
2. Splits into temporal train/test sets (80%/20%)
3. Trains neural network with JAX backend
4. Saves model parameters to `../models/`
5. Evaluates all models (Neural Network, Square-Root Law, Propagator Model)
6. Prints performance comparison

**Key Parameters** (modify in `ml.py`):
```python
layer_sizes = [input_dim, 64, 32, 16, 1]  # Network architecture
learning_rate = 0.0005                    # Initial learning rate
num_epochs = 1000                         # Training epochs
batch_size = 1024                         # Mini-batch size
```

### 4. Custom Analysis

For custom analysis or different datasets:

```python
# Example: Analyze different time horizons
import pandas as pd
from data_processing import features, target
from bouchaud_analysis import bouchaud_analysis

# Load your data
trades = pd.read_csv('your_trades_data.csv')

# Generate features
feature_df = features(trades, windows=[10, 50, 200])  # Custom windows

# Create targets with different horizons
target_1 = target(feature_df, horizon=1)   # 1-step ahead
target_10 = target(feature_df, horizon=10) # 10-step ahead

# Run theoretical analysis
prices = trades['price'].values
signs = trades['trade_sign'].values
volumes = trades['volume'].values
results = bouchaud_analysis(prices, signs, volumes, max_lag=500)
```

## Key Features and Capabilities

### Theoretical Validation
- Implements Bouchaud's propagator model with critical behavior analysis
- Tests theoretical predictions about β_c = (1-γ)/2 relationship
- Validates power law decay in sign correlations
- Compares empirical response functions with theoretical fits

### Advanced Feature Engineering
- 30+ microstructure features across multiple time scales
- Handles missing data with expanding window fallbacks
- Incorporates market regime indicators and critical behavior metrics
- Temporal consistency with proper warm-up periods

### High-Performance Computing
- JAX backend for GPU acceleration
- JIT compilation for optimized execution
- Vectorized operations for batch processing
- Memory-efficient feature computation

### Comprehensive Model Comparison
- Neural network with configurable architecture
- Traditional Square-Root Law baseline
- Theoretical propagator model implementation
- Statistical validation with multiple metrics (MSE, R²)

## Expected Results

The analysis typically reveals:

1. **Power Law Correlations**: Trade signs exhibit slow decay with γ ≈ 0.2-0.4
2. **Critical Behavior**: Response functions show signatures near β_c ≈ 0.3-0.4
3. **Volume Effects**: Larger trades show sublinear impact (concave in √volume)
4. **ML Performance**: Neural networks typically achieve 15-25% improvement over baselines
5. **Market Regimes**: Performance varies across volatility and trend conditions

## Troubleshooting

### Common Issues

**JAX Installation Problems:**
```bash
# For CPU-only installation
pip install jax jaxlib

# For CUDA 11.x
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Memory Issues:**
- Reduce batch size in `ml.py`
- Use smaller time windows in feature engineering
- Process data in chunks for large datasets

**Data Format Issues:**
- Ensure CSV files have proper headers
- Check timestamp formats (nanoseconds expected)
- Verify trade sign encoding (-1/+1 expected)

### Performance Optimization

1. **Enable GPU**: Install CUDA-compatible JAX version
2. **Increase Batch Size**: Use larger batches on high-memory systems  
3. **Reduce Features**: Select most important features to speed training
4. **Parallel Processing**: Use multiple CPU cores for data processing

## Contributing

To extend the project:

1. **New Features**: Add feature functions to `data_processing.py`
2. **Models**: Implement new baselines in `ml.py`
3. **Analysis**: Add theoretical tests to `bouchaud_analysis.py`
4. **Data Sources**: Extend data collection for other exchanges/assets

## Citation

If you use this code in research, please cite:

```
Bouchaud, J.P., Gefen, Y., Potters, M., & Wyart, M. (2004). 
Fluctuations and response in financial markets: the subtle nature of 'random' price changes. 
Quantitative Finance, 4(2), 176-190.
```