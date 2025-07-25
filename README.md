# From Square-Root Law to Dynamic Trade Effects

This project aims to analyze market impact, progressing from static models to dynamic trade effects. Starting with the Square-Root Law, this would be extended to temporal effects and trade information content with an implementation of a Propagator Model implementation that was introduced by J.P. Bouchaud in "Fluctuations and response in financial markets: The subtle nature of 'random' price changes.".

The work would begin with implementing the Square-Root Law, focusing on model validation, parameter estimation, and empirical testing. The analysis will then move to temporal aspects, studying how impact evolves from temporary to permanent states, characterizing decay patterns, and measuring time series impacts. The third phase will explore trade information content by analyzing how different trade types affect impact, considering trade signs and relative sizes. Finally, the project will implement machine learning techniques to predict price impact directly, using features derived from market microstructure and comparing the predictions with traditional models like the square-root law, particularly focusing on market conditions where static models underperform.

## 1. Baseline Implementation
- Square-Root Law: ΔP = Y σ√(Q/V)
- Parameter estimation and statistical validation
- Identify systematic deviations and failure modes

## 2. Traditional Model Extensions
- Propagator Model: P(t) = ∑G(t-s)ε(s)
- Temporal impact analysis (temporary vs permanent)
- Trade information content effects

## 3. ML Implementation
- Features: Volatility metrics, volume patterns, spreads, order book imbalance, decay patterns from propagator analysis
- Target: Direct price impact ΔP prediction
- Models: Compare ML predictions vs Square-root law vs Propagator model
  
## 4. Regime-Dependent Analysis
- Identify market conditions where traditional models underperform
- Cross-validation framework across different market regimes
- Performance comparison metrics focusing on model failure cases