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

def response(prices, signs, max_lag):

    n_trades = len(prices)
    lags = np.arange(1, min(max_lag + 1, n_trades // 2))
    response = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        price_diffs = prices[lag:] - prices[:-lag]
        trade_signs = signs[:-lag]
        
        response[i] = np.mean(price_diffs * trade_signs)
    
    return lags, response


max_lag = 1000

lags, response_func = response(prices, signs, max_lag)

plt.figure(figsize=(10, 6))
plt.loglog(lags, response_func, 'ko-', linewidth=2, markersize=4, label='R(ℓ)')
plt.xlabel('Time (Trades)', fontsize=12)
plt.ylabel('R(ℓ)', fontsize=12)
plt.title('Response Function R(ℓ)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

def conditioned_response(prices, signs, volumes, log_vols, lags, n_vol_bins, vol_percentiles):

    n_trades = len(prices)

    # Log-spaced volume bins
    vol_min, vol_max = np.percentile(log_vols, vol_percentiles)
    vol_bins = np.logspace(vol_min, vol_max, n_vol_bins + 1, base=np.e)

    response_matrix = np.full((len(lags), n_vol_bins), np.nan)
    vol_centers = np.zeros(n_vol_bins)    

    for vol_idx in range(n_vol_bins):
        # Find trades in this volume bin
        vol_mask = (volumes >= vol_bins[vol_idx]) & (volumes < vol_bins[vol_idx + 1])
        vol_centers[vol_idx] = np.sqrt(vol_bins[vol_idx] * vol_bins[vol_idx + 1])
        
        if np.sum(vol_mask) < 50:  # Skip bins with few trades
            continue
        
        vol_indices = np.where(vol_mask)[0]
        
        for i, lag in enumerate(lags):
            valid_indices = vol_indices[vol_indices < n_trades - lag]
            
            if len(valid_indices) < 10:
                continue
                
            # response
            price_diffs = prices[valid_indices + lag] - prices[valid_indices]
            trade_signs = signs[valid_indices]
            response_matrix[i, vol_idx] = np.mean(price_diffs * trade_signs)

    return {
        'lags': lags,
        'vol_centers': vol_centers,
        'vol_bins': vol_bins,
        'response_matrix': response_matrix,
        'log_vol_centers': np.log(vol_centers)
    }

vol_data = conditioned_response(prices, signs, volumes, log_volumes, lags, n_vol_bins=8, vol_percentiles=(5, 95))

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(vol_data['vol_centers'])))

for vol_idx, (vol_center, color) in enumerate(zip(vol_data['vol_centers'], colors)):
    response_vol = vol_data['response_matrix'][:, vol_idx]
    valid_mask = ~np.isnan(response_vol)
    
    if np.sum(valid_mask) > 10:
        log_vol = np.log(vol_center)
        label = f'log V=[{log_vol:.1f}]'
        
        scaled_response = response_vol / np.log(vol_center)
        
        plt.loglog(lags[valid_mask], scaled_response[valid_mask], 
                    'o-', color=color, linewidth=1.5, markersize=3,
                    alpha=0.7, label=label) 

plt.loglog(lags, response_func, 'k-', linewidth=3, label='R(ℓ) Average', alpha=0.9)

plt.xlabel('Time (Trades)', fontsize=12)
plt.ylabel('R(ℓ,V) / ln(V)', fontsize=12)
plt.title('Volume Conditioned Response R(ℓ,V)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()


def sign_correlation(signs, log_vols, lags):

    mean_log_vol = np.mean(log_vols)
    mean_sign = np.mean(signs)
    mean_sign_squared = mean_sign**2

    C0 = np.zeros(len(lags))  # Basic sign correlation
    C1 = np.zeros(len(lags))  # Cross correlation with volume
    C2 = np.zeros(len(lags))  # Volume-weighted correlation

    for i, lag in enumerate(lags):
        print(f"  Lag {lag}/{max_lag}")
            
        idx_present = slice(None, -lag)  # [0, 1, ..., n-lag-1]  
        idx_future = slice(lag, None)    # [lag, lag+1, ..., n-1]
        
        signs_present = signs[idx_present]
        signs_future = signs[idx_future] 
        log_vols_present = log_vols[idx_present]
        log_vols_future = log_vols[idx_future]
        
        # C0(l) = <ε_{n+l} ε_n> - <ε_n>²
        C0[i] = np.mean(signs_future * signs_present) - mean_sign_squared
        
        # C1(l) = <ε_{n+l} ε_n ln V_n>
        C1[i] = np.mean(signs_future * signs_present * log_vols_present)
        
        # C2(l) = <ε_{n+l} ln V_{n+l} ε_n ln V_n>
        C2[i] = np.mean(signs_future * log_vols_future * signs_present * log_vols_present)

    return {
        'lags': lags,
        'C0': C0,
        'C1': C1, 
        'C2': C2,
        'mean_log_vol': mean_log_vol
    }

def power_law(x: np.ndarray, A: float, gamma: float) -> np.ndarray:
    return A * np.power(x, -gamma)

correlations = sign_correlation(signs, log_volumes, lags)

min_lag = 10

mask = (correlations['lags'] >= min_lag) & (correlations['lags'] <= max_lag) & (correlations['C0'] > 0)

# Linear fit: log(y) = log(A) - gamma * log(x)
if np.sum(mask) > 5:
    log_lags = np.log(lags[mask])
    log_corr = np.log(correlations['C0'][mask])
    
    coeffs = np.polyfit(log_lags, log_corr, 1)
    gamma = -coeffs[0]
    A = np.exp(coeffs[1])
    
    print(f"Ajustement C₀(ℓ) ∝ ℓ^(-{gamma:.3f})")

# Simple correlation functions plot
fig, ax = plt.subplots(figsize=(10, 8))

lags = correlations['lags']
C0 = correlations['C0'] 
C1 = correlations['C1']
C2 = correlations['C2']
mean_log_vol = correlations['mean_log_vol']

ax.loglog(lags, C0, 'o-', label='C₀(ℓ)', color='blue', markersize=3, linewidth=1.5)
ax.loglog(lags, np.abs(C1), 's-', label='C₁(ℓ)', color='red', markersize=3, linewidth=1.5)  
ax.loglog(lags, C2, '^-', label='C₂(ℓ)', color='green', markersize=3, linewidth=1.5)

theoretical_C1 = mean_log_vol * C0
theoretical_C2 = mean_log_vol**2 * C0

valid_mask = C0 > 0
ax.loglog(lags[valid_mask], np.abs(theoretical_C1[valid_mask]), ':', 
         color='red', alpha=0.7, linewidth=2, label='⟨ln V⟩ C₀(ℓ)')
ax.loglog(lags[valid_mask], theoretical_C2[valid_mask], ':', 
         color='green', alpha=0.7, linewidth=2, label='⟨ln V⟩² C₀(ℓ)')

# Labels and formatting
ax.set_xlabel('Time (trades)', fontsize=14)
ax.set_ylabel('C(ℓ)', fontsize=14) 
ax.set_title('Sign Correlation Functions (Bouchaud Analysis)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xlim(1, max(lags))

textstr = f'⟨ln V⟩ = {mean_log_vol:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# ## 2. Propagator Model
# - Propagator Model: P(t) = ∑G(t-s)ε(s)
# - Temporal impact analysis (temporary vs permanent)
# - Trade information content effects

gamma = 0.24      # Sign correlation decay exponent
C0 = 0.20         # Correlation amplitude  
Gamma0 = 2.8e-3   # Overall scaling
l0 = 20           # Short-time cutoff (trades)

# Critical exponent for diffusive prices
beta_c = (1 - gamma) / 2
print(f"Critical exponent βc = {beta_c:.3f}")

# P(t) = Σ G₀(t-s) ε(s) ln(V(s))
def propagator(t, t0=20, beta=0.4, Gamma0=0.001):
    """G₀(t) = Γ₀ * t₀^β / (t₀ + t)^β"""
    return Gamma0 * (t0**beta) / ((t0 + t)**beta)

# Theoretical response function
def theo_response(lags, C0, gamma, t0=20, beta=0.4, Gamma0=0.001):
    """R(ℓ) using Eq. 17 from the paper"""
    R_theory = []
    
    for lag in lags:
        # Direct term
        term1 = propagator(lag, t0, beta, Gamma0)
        
        # Correlation terms
        term2 = 0
        for n in range(1, min(lag, 100)):
            
            if n-1 < len(correlations['C0']):
                corr = correlations['C0'][n-1]  # Use calculated C0(n)
            else:
                corr = C0 / (n**gamma)

            term2 += propagator(lag - n, t0, beta, Gamma0) * corr
        
        R_theory.append(term1 + term2)
    
    return np.array(R_theory)

def fit_propagator(params):
    t0, beta, Gamma0 = params
    if beta <= 0 or beta >= 1 or t0 <= 0 or Gamma0 <= 0:
        return 1e6
    
    R_pred = theo_response(lags[:len(response_func)], C0, gamma, t0, beta, Gamma0)
    return np.sum((response_func - R_pred)**2)

beta_values = [0.36, 0.38, 0.40, 0.42, 0.44]  # Around βc
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Calculate for different beta values using the existing method
responses = {}
plt.figure(figsize=(12, 8))

for i, beta in enumerate(beta_values):
    print(f"β = {beta}:")
    R = theo_response(lags, C0, gamma, l0, beta, Gamma0)
    responses[beta] = R
    
    # Plot each response function
    plt.loglog(lags, R, '--', color=colors[i % len(colors)], 
               linewidth=2, alpha=0.7, label=f'Theory β={beta:.2f}')

# Plot the empirical response function for comparison  
plt.loglog(lags, response_func, 'ko-', linewidth=3, markersize=4, 
           label='Empirical R(ℓ)', alpha=0.8)

# Highlight the critical beta
plt.loglog(lags, responses[beta_c], 'b-', linewidth=4, 
           alpha=0.9, label=f'Critical β={beta_c:.3f}')

plt.xlabel('Time (Trades)', fontsize=14)
plt.ylabel('R(ℓ)', fontsize=14)
plt.title('Response Function: Theory vs Empirical', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(1, max(lags))

# Add parameter information
textstr = f'γ = {gamma:.2f}\nC₀ = {C0:.2f}\nΓ₀ = {Gamma0:.1e}\nl₀ = {l0}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Print comparison statistics
print("\nFit quality comparison:")
for beta in beta_values:
    mse = np.mean((response_func - responses[beta])**2)
    print(f"β = {beta:.2f}: MSE = {mse:.2e}")

fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

# Plot response functions for different beta values
for beta, color in zip(beta_values, colors):
    R = responses[beta]
    ax1.plot(lags, R * 1000, label=f'β={beta}', color=color, alpha=0.8, linewidth=2)

ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='~100 trades')
ax1.axvline(x=1000, color='gray', linestyle=':', alpha=0.5, label='~1000 trades')

ax1.set_xlabel('Time (Trades)', fontsize=12)
ax1.set_ylabel('R(ℓ) × 1000', fontsize=12)
ax1.set_title('Response Function for Different β Values\n(Reproducing Bouchaud Fig. 9)', fontsize=14)
ax1.set_xscale('log')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

textstr = f'''Key Insights:
• βc = {beta_c:.3f} for γ = {gamma}
• β < βc: Upward trend
• β > βc: Downward trend  
• β ≈ βc: Nearly constant'''

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# Calculate propagator for extended range
lags_extended = np.arange(1, 10000)
G0_fit = np.array([propagator(l, 0.42) for l in lags_extended])
G0_critical = np.array([propagator(l, beta_c) for l in lags_extended])

# Plot on log-log scale
ax2.loglog(lags_extended, G0_fit, label='G₀ fit (β=0.42)', color='green', linewidth=2)
ax2.loglog(lags_extended, G0_critical, label=f'G₀ critical (β={beta_c:.3f})', 
           color='red', linestyle='--', linewidth=2)

# Add power-law reference lines
ax2.loglog(lags_extended, 1e-3 * lags_extended**(-0.42), 'k:', alpha=0.5, 
           label='l^(-0.42)')
ax2.loglog(lags_extended, 1e-3 * lags_extended**(-beta_c), 'k--', alpha=0.5, 
           label=f'l^(-{beta_c:.2f})')

ax2.set_xlabel('ℓ (trades)', fontsize=12)
ax2.set_ylabel('G₀(ℓ)', fontsize=12)
ax2.set_title('Bare Propagator G₀(ℓ)\n(Reproducing Bouchaud Fig. 10)', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print(f"\nModel Parameters:")
print(f"• Sign correlation decay: γ = {gamma}")
print(f"• Correlation amplitude: C₀ = {C0}")
print(f"• Overall scaling: Γ₀ = {Gamma0}")
print(f"• Short-time cutoff: ℓ₀ = {l0}")

print(f"• Critical exponent: βc = (1-γ)/2 = {beta_c:.3f}")
print(f"• Critical condition: 2β + γ = {2*beta_c + gamma:.1f}")

print(f"• For β = {beta_c:.3f} (critical): R(ℓ) nearly constant")
print(f"• For β = 0.42 (best fit): R(ℓ) shows realistic max/decay")

print(f"\nPhysical Interpretation:")
print(f"• Market at critical point balancing:")
print(f"  - Long-range correlations (super-diffusion)")
print(f"  - Impact decay (sub-diffusion)")
print(f"• Result: Diffusive (random walk) price process")

R_max_idx = np.argmax(responses[0.42])
R_max_lag = lags[R_max_idx]
R_max_value = responses[0.42][R_max_idx]

print(f"• Maximum response at ℓ ≈ {R_max_lag} trades")
print(f"• Maximum value: R_max = {R_max_value*1000:.3f} × 10⁻³")
print(f"• Ratio R_max/R(1) = {R_max_value/responses[0.42][0]:.2f}")
