import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional, Tuple

def fit_loess_bootstrap(
    ts: pd.Series,
    frac: float = 0.7,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.85,
    random_state: Optional[int] = 42
) -> dict:
    """Fit LOESS smoother and calculate bootstrap confidence intervals for time series.
    
    Args:
        ts: Time series with datetime index
        frac: Fraction of data to use for each local regression (0.1 = tight, 0.7 = smooth)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals (0.95 = 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        dict with keys: 'fitted', 'residuals', 'ci_lower', 'ci_upper', 'bootstrap_samples'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert datetime index to numeric (days since start)
    start_date = ts.index.min()
    x_numeric = np.array([(date - start_date).days for date in ts.index])
    y_values = ts.values
    
    # Fit LOESS
    fitted_values = lowess(y_values, x_numeric, frac=frac, return_sorted=False)
    residuals = y_values - fitted_values
    
    # Bootstrap residuals
    n_points = len(residuals)
    bootstrap_samples = np.zeros((n_bootstrap, n_points))
    
    for i in range(n_bootstrap):
        boot_residuals = np.random.choice(residuals, size=n_points, replace=True)
        bootstrap_samples[i] = fitted_values + boot_residuals
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_samples, 100 * alpha/2, axis=0)
    ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha/2), axis=0)
    
    return {
        'fitted': fitted_values,
        'residuals': residuals,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_samples': bootstrap_samples
    }

def plot_time_series_with_ci(
    ts: pd.Series,
    loess_results: dict,
    title: str,
    ylabel: str = "Value",
    reference_date: Optional[str] = "2018-01-01",
    reference_label: str = "Jan 2018",
    figsize: Tuple[int, int] = (12, 6),
    show_points: bool = True
) -> plt.Figure:
    """Plot time series with LOESS trend and confidence intervals.
    
    Args:
        ts: Original time series
        loess_results: Results from fit_loess_bootstrap()
        title: Plot title
        ylabel: Y-axis label
        reference_date: Date for vertical reference line
        reference_label: Label for reference line
        figsize: Figure size
        show_points: Whether to show individual data points
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original data
    if show_points:
        ax.plot(ts.index, ts.values, 'o-', color='steelblue', 
               alpha=0.7, label='Observed Data', markersize=4)
    else:
        ax.plot(ts.index, ts.values, '-', color='steelblue', 
               alpha=0.7, label='Observed Data', linewidth=1)
    
    # Plot LOESS fit
    ax.plot(ts.index, loess_results['fitted'], '-', color='black', 
           linewidth=2, label='LOESS Trend')
    
    # Plot confidence bands
    ax.fill_between(ts.index, 
                   loess_results['ci_lower'], 
                   loess_results['ci_upper'],
                   alpha=0.3, color='grey', label='80% Bootstrap CI')
    
    # Add reference line if specified
    if reference_date:
        ax.axvline(x=pd.to_datetime(reference_date), color='red', 
                  linestyle='--', alpha=0.7, label=reference_label)
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    
    return fig

def analyze_trend_change(
    ts: pd.Series,
    loess_results: dict,
    change_date: str = "2018-01-01",
    print_results: bool = True
) -> dict:
    """Analyze trend change before/after a specific date.
    
    Args:
        ts: Time series
        loess_results: Results from fit_loess_bootstrap()
        change_date: Date to analyze change around
        print_results: Whether to print summary statistics
        
    Returns:
        dict with trend change statistics
    """
    change_dt = pd.to_datetime(change_date)
    
    # Find index of change date
    try:
        change_idx = list(ts.index).index(change_dt)
    except ValueError:
        # If exact date not found, find closest
        change_idx = np.argmin(np.abs(ts.index - change_dt))
    
    fitted = loess_results['fitted']
    residuals = loess_results['residuals']
    ci_lower = loess_results['ci_lower']
    ci_upper = loess_results['ci_upper']
    
    # Calculate statistics
    pre_trend = np.mean(fitted[:change_idx])
    post_trend = np.mean(fitted[change_idx:])
    trend_change = post_trend - pre_trend
    
    pre_volatility = np.std(residuals[:change_idx])
    post_volatility = np.std(residuals[change_idx:])
    
    avg_ci_width = np.mean(ci_upper - ci_lower)
    
    results = {
        'pre_trend_avg': pre_trend,
        'post_trend_avg': post_trend,
        'trend_change': trend_change,
        'pre_volatility': pre_volatility,
        'post_volatility': post_volatility,
        'avg_ci_width': avg_ci_width,
        'change_date': change_date,
        'n_pre': change_idx,
        'n_post': len(ts) - change_idx
    }
    
    if print_results:
        print(f"Trend Analysis around {change_date}:")
        print(f"  Pre-period average: {pre_trend:.1f}")
        print(f"  Post-period average: {post_trend:.1f}")
        print(f"  Change: {trend_change:+.1f}")
        print(f"  Pre-period volatility: {pre_volatility:.1f}")
        print(f"  Post-period volatility: {post_volatility:.1f}")
        print(f"  Average CI width: {avg_ci_width:.1f}")
        print(f"  Trend change magnitude vs CI width: {abs(trend_change)/avg_ci_width:.2f}")
    
    return results

# Convenience function for complete analysis
def analyze_time_series(
    ts: pd.Series,
    title: str,
    ylabel: str = "Value",
    frac: float = 0.3,
    reference_date: str = "2018-01-01",
    **kwargs
) -> Tuple[dict, plt.Figure]:
    """Complete time series analysis: LOESS + bootstrap + plotting + trend analysis.
    
    Args:
        ts: Time series
        title: Plot title
        ylabel: Y-axis label
        frac: LOESS smoothing fraction
        reference_date: Date for trend change analysis
        **kwargs: Additional arguments passed to fit_loess_bootstrap()
    
    Returns:
        tuple: (analysis_results, figure)
    """
    # Fit LOESS and bootstrap
    loess_results = fit_loess_bootstrap(ts, frac=frac, **kwargs)
    
    # Create plot
    fig = plot_time_series_with_ci(ts, loess_results, title, ylabel, reference_date)
    
    # Analyze trend change
    trend_analysis = analyze_trend_change(ts, loess_results, reference_date)
    
    # Combine results
    analysis_results = {
        'loess': loess_results,
        'trend_change': trend_analysis
    }
    
    return analysis_results, fig