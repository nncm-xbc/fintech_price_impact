#fintech
# Plan: 

### 1. Square-Root Law Implementation
	- Base implementation: ΔP = Y σ√(Q/V)
	- Critical assumptions and limitations
	- Data cleaning and preparation methodology
	- Parameter estimation and model fit analysis
	- Statistical validation of the law

### 2. Temporal Impact: From Static to Dynamic
	- Studies how price impact evolves over time
	- Differentiates between:
	    - Temporary impact (immediate price reaction)
	    - Permanent impact (long-term price change)
	    - Decay patterns between these states
	- Uses time series analysis to measure impact persistence

### 3. Information Content of Trades
	- Analyzes how different trade types affect impact
	- Considers:
	    - Trade sign (buy/sell)
	    - Trade size relative to normal volume
	- Link between information content and decay patterns
	- Volume-synchronized measures

### 4. ML integration
- Feature Construction: 
	- Market volatility metrics 
	- Trading volume patterns 
	- Spread measurements 
	- Order book imbalance 
-  Prediction Targets: 
	- Direct price impact ΔP 
	- Compare ML predictions vs Square-root law predictions
	- Focus on cases where square-root law under-performs 
- Implementation: 
	- Model selection and training 
	- Cross-validation framework 
	- Comparisons of metrics during different market regime



From Square-Root Law to Dynamic Trade Effects

This project would aim to analyze market impact, progressing from static models to dynamic trade effects. Starting with the Square-Root Law, this would be extended to temporal effects and trade information content with an implementation of a Propagator Model implementation that was introduced by J.P. Bouchaud in "Fluctuations and response in financial markets: The subtle nature of 'random' price changes.".

The work would begin with implementing the Square-Root Law, focusing on model validation, parameter estimation, and empirical testing. The analysis will then move to temporal aspects, studying how impact evolves from temporary to permanent states, characterizing decay patterns, and measuring time series impacts. The third phase will explore trade information content by analyzing how different trade types affect impact, considering trade signs and relative sizes. Finally, the project will implement machine learning techniques to predict price impact directly, using features derived from market microstructure and comparing the predictions with traditional models like the square-root law, particularly focusing on market conditions where static models underperform.

General plan:

1. Square-Root Law Implementation

- Model validation and parameter estimation
- Statistical analysis and empirical testing
- Critical assumptions examination

2. Temporal Impact Analysis

- Evolution from temporary to permanent impact
- Decay pattern characterization
- Time series impact measurement

3. Trade Information Content

- Analysis of different trade types' impact
- Trade sign (buy/sell) effects
- Trade size relative to normal volume
- Link between information content and decay patterns

4. Propagator Model Implementation

- Dynamic price response modeling: P(t) = ∑G(t-s)ε(s)
- Kernel estimation and memory effects
- Comparison with static model