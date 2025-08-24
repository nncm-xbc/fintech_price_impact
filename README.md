<div align="center">

# **Beyond Static Models in Cryptocurrency Markets**

[Installation](#installation) • [Usage](#usage)

</div>
This project implements a comprehensive analysis of market impact dynamics, progressing from traditional static models to advanced temporal effects and machine learning approaches. The work applies the propagator model framework introduced by J.P. Bouchaud in "Fluctuations and response in financial markets: The subtle nature of 'random' price changes" to Bitcoin (BTCUSDT) high-frequency trading data, extending the original equity market analysis to cryptocurrency markets.

## Project Overview

The project analyzes market impact through four main phases:

1. **Baseline Implementation**: Square-Root Law (ΔP = Y σ√(Q/V)) with statistical validation
2. **Traditional Model Extensions**: Propagator Model (P(t) = ∑G(t-s)ε(s)) and temporal impact analysis
3. **Machine Learning Implementation**: Neural network prediction of price impact using microstructure features
4. **Regime-Dependent Analysis**: Cross-validation across different market conditions

## Data Sources

The project uses high-frequency cryptocurrency trading data from Binance vision:

- **Trade Data**: Individual BTCUSDT trades with nanosecond timestamps
  - Files: `BTCUSDT-trades*.csv` in `data/binance_raw/`
  - Contains: trade_id, price, volume, quote_volume, timestamp, is_buyer_maker, is_best_match
  
   https://data.binance.vision/?prefix=data/spot/daily/trades/BTCUSDT/

- **Quote Data**: 1-second OHLCV candle data
  - Files: `BTCUSDT-1s*.csv` in `data/binance_raw/`
  - Contains: open_time, open, high, low, close, volume, quote_volume, count
  
   https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1s/

## Installation

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

## Usage

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

## Citation
This work has been greatly based on the following paper, if you use this code in research, please cite:
```
Bouchaud, J.P., Gefen, Y., Potters, M., & Wyart, M. (2004). 
Fluctuations and response in financial markets: the subtle nature of 'random' price changes. 
Quantitative Finance, 4(2), 176-190.
```