import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


#==================================================================================
# Response Functions
#==================================================================================

def response(prices, signs, max_lag=1000):

    prices = np.array(prices)
    signs = np.array(signs)
    n_trades = len(prices)
    max_lag = min(max_lag, n_trades // 2)
    
    lags = np.arange(1, max_lag + 1)
    response = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        if lag >= n_trades:
            break
        price_diffs = prices[lag:] - prices[:-lag]
        trade_signs = signs[:-lag]
        response[i] = np.mean(price_diffs * trade_signs)
    
    # Return only valid data
    valid_idx = np.isfinite(response) & (response != 0)
    return {
        'lags': lags[valid_idx],
        'values': response[valid_idx]
    }


def conditioned_response(prices, signs, volumes, lags, n_vol_bins=8, vol_percentiles=(5, 95)):

    prices = np.array(prices)
    signs = np.array(signs)
    volumes = np.array(volumes)
    log_volumes = np.log(volumes)
    n_trades = len(prices)
    
    # Create log-spaced volume bins
    vol_min, vol_max = np.percentile(log_volumes, vol_percentiles)
    vol_bins = np.logspace(vol_min, vol_max, n_vol_bins + 1, base=np.e)
    vol_centers = np.sqrt(vol_bins[:-1] * vol_bins[1:])
    
    response_matrix = np.full((len(lags), n_vol_bins), np.nan)
    
    for vol_idx in range(n_vol_bins):
        vol_mask = ((volumes >= vol_bins[vol_idx]) & 
                   (volumes < vol_bins[vol_idx + 1]))
        
        if np.sum(vol_mask) < 50:  # Skip bins with few trades
            continue
            
        vol_indices = np.where(vol_mask)[0]
        
        for i, lag in enumerate(lags):
            valid_indices = vol_indices[vol_indices < n_trades - lag]
            
            if len(valid_indices) < 10:
                continue
            
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

#==================================================================================
# Sign Correlation Functions
#==================================================================================

def sign_correlations(signs, volumes, lags):

    signs = np.array(signs)
    log_volumes = np.log(volumes)
    mean_log_vol = np.mean(log_volumes)
    mean_sign = np.mean(signs)
    mean_sign_squared = mean_sign**2
    
    C0 = np.zeros(len(lags))
    C1 = np.zeros(len(lags))
    C2 = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        idx_present = slice(None, -lag)
        idx_future = slice(lag, None)
        
        signs_present = signs[idx_present]
        signs_future = signs[idx_future]
        log_vols_present = log_volumes[idx_present]
        log_vols_future = log_volumes[idx_future]
        
        # C₀(ℓ) = ⟨ε_{n+ℓ} ε_n⟩ - ⟨ε_n⟩²
        C0[i] = np.mean(signs_future * signs_present) - mean_sign_squared
        
        # C₁(ℓ) = ⟨ε_{n+ℓ} ε_n ln V_n⟩
        C1[i] = np.mean(signs_future * signs_present * log_vols_present)
        
        # C₂(ℓ) = ⟨ε_{n+ℓ} ln V_{n+ℓ} ε_n ln V_n⟩
        C2[i] = np.mean(signs_future * log_vols_future * signs_present * log_vols_present)
    
    return {
        'lags': lags,
        'C0': C0,
        'C1': C1,
        'C2': C2,
        'mean_log_vol': mean_log_vol
    }


def power_law(lags, C0, min_lag=10):

    lags = np.array(lags)
    C0 = np.array(C0)
    
    # Use only positive correlations above min_lag
    mask = (lags >= min_lag) & (C0 > 0)
    
    if np.sum(mask) < 5:
        print("Warning: Insufficient data for power law fit")
        return None
    
    # Linear fit in log space: log(C₀) = log(A) - γ*log(ℓ)
    log_lags = np.log(lags[mask])
    log_C0 = np.log(C0[mask])
    
    coeffs = np.polyfit(log_lags, log_C0, 1)
    gamma = -coeffs[0]
    A = np.exp(coeffs[1])
    
    # Calculate R²
    predicted = A * lags[mask]**(-gamma)
    r2 = r2_score(C0[mask], predicted)
    
    return {
        'gamma': gamma,
        'amplitude': A,
        'r2': r2,
        'min_lag': min_lag
    }

#==================================================================================
# Propagator Model
#==================================================================================

def propagator_function(t, t0=20, beta=0.4, Gamma0=0.001):
    # G₀(t) = Γ₀ * t₀^β / (t₀ + t)^β
    
    return Gamma0 * (t0**beta) / ((t0 + t)**beta)


def theoretical_response(lags, gamma, C0_amp, t0=20, beta=0.4, Gamma0=0.001):

    R_theory = []
    
    for lag in lags:
        # Direct term
        term1 = propagator_function(lag, t0, beta, Gamma0)
        
        # Correlation sum (truncated for performance)
        term2 = 0
        max_n = min(lag, 100)
        
        for n in range(1, max_n):
            # Use power law approximation for correlations
            corr = C0_amp / (n**gamma) if n > 1 else C0_amp
            term2 += propagator_function(lag - n, t0, beta, Gamma0) * corr
        
        R_theory.append(term1 + term2)
    
    return np.array(R_theory)


def analyze_critical_behavior(response_data, correlation_data, power_law_data, beta_range=None):

    gamma = power_law_data['gamma']
    C0_amp = power_law_data['amplitude']
    beta_c = (1 - gamma) / 2
    
    if beta_range is None:
        beta_range = np.arange(beta_c - 0.04, beta_c + 0.06, 0.02)
    
    lags = response_data['lags']
    response_emp = response_data['values']
    
    # Calculate theoretical responses for different β values
    theoretical_responses = {}
    fit_scores = {}
    
    for beta in beta_range:
        R_theory = theoretical_response(lags, gamma, C0_amp, beta=beta)
        theoretical_responses[beta] = R_theory
        
        # Calculate fit quality (MSE)
        valid_mask = np.isfinite(R_theory) & np.isfinite(response_emp)
        if np.sum(valid_mask) > 0:
            fit_scores[beta] = mean_squared_error(
                response_emp[valid_mask], R_theory[valid_mask]
            )
    
    # Find best fit
    best_beta = min(fit_scores.keys(), key=lambda x: fit_scores[x])
    
    return {
        'gamma': gamma,
        'beta_c': beta_c,
        'best_beta': best_beta,
        'beta_range': beta_range,
        'theoretical_responses': theoretical_responses,
        'fit_scores': fit_scores
    }

#==================================================================================
# Plotting Functions
#==================================================================================

def plot_response_function(response_data, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(response_data['lags'], response_data['values'], 'ko-', 
             linewidth=2, markersize=4, label='R(ℓ)')
    
    ax.set_xlabel('Time (Trades)')
    ax.set_ylabel('R(ℓ)')
    ax.set_title('Response Function R(ℓ)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_volume_conditioned_response(volume_response_data, response_data, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(volume_response_data['vol_centers'])))
    
    # Plot volume-conditioned responses
    for vol_idx, (vol_center, color) in enumerate(zip(volume_response_data['vol_centers'], colors)):
        response_vol = volume_response_data['response_matrix'][:, vol_idx]
        valid_mask = ~np.isnan(response_vol)
        
        if np.sum(valid_mask) > 10:
            log_vol = np.log(vol_center)
            label = f'log V = {log_vol:.1f}'
            
            # Scale by log volume
            scaled_response = response_vol / np.log(vol_center)
            
            ax.loglog(volume_response_data['lags'][valid_mask], scaled_response[valid_mask],
                     'o-', color=color, linewidth=1.5, markersize=3,
                     alpha=0.7, label=label)
    
    # Plot average response
    ax.loglog(response_data['lags'], response_data['values'], 'k-', 
             linewidth=3, label='R(ℓ) Average', alpha=0.9)
    
    ax.set_xlabel('Time (Trades)')
    ax.set_ylabel('R(ℓ,V) / ln(V)')
    ax.set_title('Volume Conditioned Response R(ℓ,V)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_correlation_functions(correlation_data, power_law_data=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    lags = correlation_data['lags']
    C0, C1, C2 = correlation_data['C0'], correlation_data['C1'], correlation_data['C2']
    mean_log_vol = correlation_data['mean_log_vol']
    
    # Plot correlation functions
    ax.loglog(lags, C0, 'o-', label='C₀(ℓ)', color='blue', 
             markersize=3, linewidth=1.5)
    ax.loglog(lags, np.abs(C1), 's-', label='|C₁(ℓ)|', color='red', 
             markersize=3, linewidth=1.5)
    ax.loglog(lags, C2, '^-', label='C₂(ℓ)', color='green', 
             markersize=3, linewidth=1.5)
    
    # Plot theoretical power law fit
    if power_law_data is not None:
        gamma = power_law_data['gamma']
        theoretical_fit = power_law_data['amplitude'] * lags**(-gamma)
        ax.loglog(lags, theoretical_fit, '--', color='blue', alpha=0.7, 
                 linewidth=2, label=f'ℓ^(-{gamma:.2f})')
    
    # Theoretical C1 and C2
    theoretical_C1 = mean_log_vol * C0
    theoretical_C2 = mean_log_vol**2 * C0
    
    valid_mask = C0 > 0
    ax.loglog(lags[valid_mask], np.abs(theoretical_C1[valid_mask]), ':', 
             color='red', alpha=0.7, linewidth=2, label='⟨ln V⟩ C₀(ℓ)')
    ax.loglog(lags[valid_mask], theoretical_C2[valid_mask], ':', 
             color='green', alpha=0.7, linewidth=2, label='⟨ln V⟩² C₀(ℓ)')
    
    ax.set_xlabel('Time (trades)')
    ax.set_ylabel('C(ℓ)')
    ax.set_title('Sign Correlation Functions')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add parameter info
    if power_law_data is not None:
        textstr = f'γ = {power_law_data["gamma"]:.3f}\n⟨ln V⟩ = {mean_log_vol:.2f}\nR² = {power_law_data["r2"]:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_critical_analysis(critical_data, response_data, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot response functions for different β values
    colors = plt.cm.viridis(np.linspace(0, 1, len(critical_data['beta_range'])))
    
    for beta, color in zip(critical_data['beta_range'], colors):
        if beta in critical_data['theoretical_responses']:
            R_theory = critical_data['theoretical_responses'][beta]
            ax1.loglog(response_data['lags'], R_theory, '--', color=color, 
                      linewidth=2, alpha=0.7, label=f'β = {beta:.3f}')
    
    # Highlight critical and best fit
    if critical_data['beta_c'] in critical_data['theoretical_responses']:
        R_critical = critical_data['theoretical_responses'][critical_data['beta_c']]
        ax1.loglog(response_data['lags'], R_critical, 'b-', linewidth=3, 
                  alpha=0.9, label=f'Critical β = {critical_data["beta_c"]:.3f}')
    
    # Plot empirical
    ax1.loglog(response_data['lags'], response_data['values'], 'ko-', 
              linewidth=2, markersize=4, label='Empirical R(ℓ)', alpha=0.8)
    
    ax1.set_xlabel('Time (Trades)')
    ax1.set_ylabel('R(ℓ)')
    ax1.set_title('Response Function: Theory vs Empirical')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot fit quality
    betas = list(critical_data['fit_scores'].keys())
    scores = list(critical_data['fit_scores'].values())
    
    ax2.semilogy(betas, scores, 'o-', linewidth=2, markersize=6)
    ax2.axvline(critical_data['beta_c'], color='red', linestyle='--', 
               label=f'Critical β = {critical_data["beta_c"]:.3f}')
    ax2.axvline(critical_data['best_beta'], color='green', linestyle='--', 
               label=f'Best fit β = {critical_data["best_beta"]:.3f}')
    
    ax2.set_xlabel('β')
    ax2.set_ylabel('MSE')
    ax2.set_title('Fit Quality vs β')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

#==================================================================================
# Summary and Main Analysis
#==================================================================================

def print_analysis_summary(response_data, correlation_data, power_law_data, critical_data, 
                          prices, volumes):
    """Print comprehensive analysis summary."""
    print("=" * 60)
    print("BOUCHAUD MARKET MICROSTRUCTURE ANALYSIS")
    print("=" * 60)
    
    print(f"\nData Summary:")
    print(f"• Total trades: {len(prices):,}")
    print(f"• Analysis range: 1 - {max(response_data['lags'])} lags")
    print(f"• Mean log volume: {np.mean(np.log(volumes)):.2f}")
    
    if power_law_data is not None:
        print(f"\nSign Correlation Power Law:")
        print(f"• Decay exponent γ = {power_law_data['gamma']:.3f}")
        print(f"• Amplitude A = {power_law_data['amplitude']:.3f}")
        print(f"• R² = {power_law_data['r2']:.3f}")
    
    if critical_data is not None:
        print(f"\nCritical Behavior Analysis:")
        print(f"• Critical exponent βc = (1-γ)/2 = {critical_data['beta_c']:.3f}")
        print(f"• Best fit β = {critical_data['best_beta']:.3f}")
        print(f"• Critical condition: 2β + γ = {2*critical_data['beta_c'] + critical_data['gamma']:.1f}")
        
        best_mse = critical_data['fit_scores'][critical_data['best_beta']]
        print(f"• Best fit MSE = {best_mse:.2e}")
    
    print(f"\nPhysical Interpretation:")
    print(f"• Market exhibits long-range memory in trade signs")
    print(f"• Price response shows temporary-to-permanent transition")
    print(f"• Critical balance between correlation and impact decay")
    print(f"• Results consistent with diffusive price process")
    print("=" * 60)


def bouchaud_analysis(prices, signs, volumes, max_lag=1000, plot=True, save_plots=False):

    print("Running Bouchaud analysis...")
    
    # Core calculations
    print("• Computing response function...")
    response_data = response(prices, signs, max_lag)
    
    print("• Computing volume-conditioned response...")
    volume_response_data = conditioned_response(
        prices, signs, volumes, response_data['lags']
    )
    
    print("• Computing sign correlations...")
    correlation_data = sign_correlations(signs, volumes, response_data['lags'])
    
    print("• Fitting power law...")
    power_law_data = power_law(correlation_data['lags'], correlation_data['C0'])
    
    print("• Analyzing critical behavior...")
    critical_data = None
    if power_law_data is not None:
        critical_data = analyze_critical_behavior(
            response_data, correlation_data, power_law_data
        )
    
    # Generate plots
    if plot:
        print("• Generating plots...")
        plot_response_function(response_data)
        plot_volume_conditioned_response(volume_response_data, response_data)
        plot_correlation_functions(correlation_data, power_law_data)
        if critical_data is not None:
            plot_critical_analysis(critical_data, response_data)
    
    # Print summary
    print_analysis_summary(response_data, correlation_data, power_law_data, 
                          critical_data, prices, volumes)
    
    # Compile results
    results = {
        'response': response_data,
        'volume_response': volume_response_data,
        'correlations': correlation_data,
        'power_law': power_law_data,
        'critical_analysis': critical_data
    }
    
    return results

if __name__ == "__main__":
    print("Bouchaud Analysis Module")
    trades = pd.read_csv('../data/trades_sparse.csv')
    prices = trades['price'].values
    signs = trades['trade_sign'].values
    volumes = trades['volume'].values
    results = bouchaud_analysis(prices, signs, volumes)
    print("Individual functions also available for modular analysis")