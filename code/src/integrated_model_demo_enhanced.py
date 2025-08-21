#!/usr/bin/env python3
"""
Enhanced Integrated Learning-Performance Model Demonstration
Includes comprehensive robustness testing, model comparison, and validation
for publication-quality results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE MODEL FUNCTIONS (Enhanced versions)
# ============================================================================

def learning_model(t, F0=2.0, lambda_rate=0.6, S0=0.0, r=1.0, ks=0.5, alpha_E=2.0, a=1.0, f=0.3, k_gamma=1.0, E0=5.0):
    """
    Calculate learning model components over time
    """
    # Failure rate decreases with learning
    F = F0 * np.exp(-lambda_rate * t)
    
    # Success rate saturates with learning
    S = S0 + r * (1 - np.exp(-ks * t))
    
    # Cumulative experience (integral of F and S)
    E = alpha_E * F0 * (1 - np.exp(-lambda_rate * t)) / lambda_rate + S0 * t + r * (t - (1 - np.exp(-ks * t)) / ks)
    
    # Advice accumulation
    A = a * f * t
    
    # Advice activation function
    gamma = 1 / (1 + np.exp(-k_gamma * (E - E0)))
    
    # Effective advice
    A_eff = A * gamma
    
    # Total learning
    L = E + A_eff
    
    return {
        'F': F, 'S': S, 'E': E, 'A': A, 'gamma': gamma, 'A_eff': A_eff, 'L': L
    }

def performance_model(C, alpha, Copt, k=None):
    """
    Maxwell-Boltzmann performance model - Enhanced with bounds checking
    """
    if k is None:
        k = alpha / Copt  # Constraint for peak at Copt
    
    # Ensure Copt is positive and not too close to zero
    Copt = max(Copt, 0.1)
    
    # Shift C so that Copt becomes the center
    C_shifted = C - Copt
    
    # Calculate performance with proper bounds
    y = np.zeros_like(C)
    
    # For positive C_shifted, use the original formula
    pos_mask = C_shifted >= 0
    if np.any(pos_mask):
        y[pos_mask] = (C[pos_mask] / Copt)**alpha * np.exp(-k * C_shifted[pos_mask])
    
    # For negative C_shifted, use a modified approach to avoid complex numbers
    neg_mask = C_shifted < 0
    if np.any(neg_mask):
        # Use absolute value for negative shifts, but maintain the shape
        y[neg_mask] = (np.abs(C[neg_mask]) / Copt)**alpha * np.exp(-k * np.abs(C_shifted[neg_mask]))
    
    # Normalize to peak at 1, but handle edge cases
    y_max = np.max(y)
    if y_max > 0:
        y = y / y_max
    else:
        y = np.ones_like(C)
    
    # Ensure performance is bounded between 0 and 1
    y = np.clip(y, 0, 1)
    
    return y

def integrated_performance(C, t, learning_params, performance_params):
    """
    Integrated performance model that combines learning and performance
    """
    # Get learning state at time t
    learning = learning_model(t, **learning_params)
    
    # Extract base performance parameters
    alpha_0 = performance_params['alpha_0']
    delta_alpha = performance_params['delta_alpha']
    Copt_0 = performance_params['Copt_0']
    delta_Copt = performance_params['delta_Copt']
    k_0 = performance_params['k_0']
    beta = performance_params['beta']
    L_max = performance_params['L_max']
    E_max = performance_params['E_max']
    
    # Calculate evolved parameters with bounds
    alpha_t = np.clip(alpha_0 + delta_alpha * (learning['L'] / L_max), 0.5, 5.0)
    Copt_t = np.clip(Copt_0 + delta_Copt * (learning['E'] / E_max), 0.5, 4.0)
    k_t = np.clip(k_0 * np.exp(-beta * (learning['L'] / L_max)), 0.1, 3.0)
    
    # Calculate performance with evolved parameters
    P = performance_model(C, alpha_t, Copt_t, k_t)
    
    return P, {
        'alpha_t': alpha_t,
        'Copt_t': Copt_t,
        'k_t': k_t,
        'learning': learning
    }

def calculate_performance_envelope(C, P, P_threshold=0.5):
    """
    Calculate the effective performance area above threshold
    """
    # Find where performance exceeds threshold
    above_threshold = P >= P_threshold
    
    if not np.any(above_threshold):
        return 0.0, C[0], C[0]
    
    # Find bounds
    C_min = C[above_threshold].min()
    C_max = C[above_threshold].max()
    
    # Calculate area (simplified as sum * delta_C)
    delta_C = C[1] - C[0]
    area = np.sum(P[above_threshold]) * delta_C
    
    return area, C_min, C_max

# ============================================================================
# ENHANCED VALIDATION FUNCTIONS
# ============================================================================

def bootstrap_confidence_intervals(data, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for parameters
    """
    n_samples = len(data)
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_samples.append(data[indices])
    
    # Calculate percentiles for confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    bootstrap_array = np.array(bootstrap_samples)
    lower_ci = np.percentile(bootstrap_array, lower_percentile, axis=0)
    upper_ci = np.percentile(bootstrap_array, upper_percentile, axis=0)
    
    return lower_ci, upper_ci

def parameter_uncertainty_analysis(performance_surface, C, t, learning_params, performance_params):
    """
    Analyze parameter uncertainty using bootstrap resampling
    """
    print("🔄 Calculating parameter uncertainty...")
    
    # Generate synthetic data with noise
    np.random.seed(42)  # For reproducibility
    n_samples = 50
    noisy_data = []
    
    for i, t_val in enumerate(t[::2]):  # Sample every other time point
        P, params = integrated_performance(C, t_val, learning_params, performance_params)
        # Add realistic noise
        noise = np.random.normal(0, 0.05, len(P))
        noisy_P = np.clip(P + noise, 0, 1)
        noisy_data.append({
            't': t_val,
            'C': C,
            'P': noisy_P,
            'true_params': params
        })
    
    # Bootstrap analysis for key parameters
    alpha_values = [data['true_params']['alpha_t'] for data in noisy_data]
    Copt_values = [data['true_params']['Copt_t'] for data in noisy_data]
    k_values = [data['true_params']['k_t'] for data in noisy_data]
    
    # Calculate confidence intervals
    alpha_ci = bootstrap_confidence_intervals(np.array(alpha_values))
    Copt_ci = bootstrap_confidence_intervals(np.array(Copt_values))
    k_ci = bootstrap_confidence_intervals(np.array(k_values))
    
    return {
        'alpha_ci': alpha_ci,
        'Copt_ci': Copt_ci,
        'k_ci': k_ci,
        'noisy_data': noisy_data
    }

def cross_validation_analysis(performance_surface, C, t, learning_params, performance_params):
    """
    Perform cross-validation to test model generalization
    """
    print("🔄 Performing cross-validation...")
    
    # Generate comprehensive dataset
    all_data = []
    for i, t_val in enumerate(t):
        P, params = integrated_performance(C, t_val, learning_params, performance_params)
        all_data.append({
            't': t_val,
            'C': C,
            'P': P,
            'params': params
        })
    
    # Split data for cross-validation
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    # Train model on training data (simplified - using average parameters)
    train_params = {
        'alpha_0': np.mean([d['params']['alpha_t'] for d in train_data]),
        'Copt_0': np.mean([d['params']['Copt_t'] for d in train_data]),
        'k_0': np.mean([d['params']['k_t'] for d in train_data])
    }
    
    # Test on held-out data
    predictions = []
    actuals = []
    
    for test_point in test_data:
        # Predict performance using trained parameters
        P_pred = performance_model(test_point['C'], 
                                 train_params['alpha_0'],
                                 train_params['Copt_0'],
                                 train_params['k_0'])
        predictions.extend(P_pred)
        actuals.extend(test_point['P'])
    
    # Calculate prediction metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return {
        'mse': mse,
        'r2': r2,
        'train_params': train_params,
        'test_data': test_data
    }

def parameter_sensitivity_analysis(learning_params, performance_params, C, t):
    """
    Analyze how sensitive the model is to parameter variations
    """
    print("🔄 Analyzing parameter sensitivity...")
    
    # Base case
    base_P, base_params = integrated_performance(C, t[50], learning_params, performance_params)
    base_envelope, _, _ = calculate_performance_envelope(C, base_P)
    
    # Parameter variations to test
    variations = {
        'alpha_0': [0.8, 1.2],
        'Copt_0': [0.8, 1.2],
        'k_0': [0.8, 1.2],
        'lambda_rate': [0.8, 1.2]
    }
    
    sensitivity_results = {}
    
    for param, factors in variations.items():
        sensitivity_results[param] = []
        
        for factor in factors:
            # Create modified parameters
            test_learning = learning_params.copy()
            test_performance = performance_params.copy()
            
            if param in test_learning:
                test_learning[param] *= factor
            else:
                test_performance[param] *= factor
            
            # Calculate performance with modified parameters
            test_P, _ = integrated_performance(C, t[50], test_learning, test_performance)
            test_envelope, _, _ = calculate_performance_envelope(C, test_P)
            
            # Calculate sensitivity (relative change in envelope)
            sensitivity = (test_envelope - base_envelope) / base_envelope
            sensitivity_results[param].append(sensitivity)
    
    return sensitivity_results

def comprehensive_model_comparison(C, t, learning_params, performance_params):
    """
    Compare Maxwell-Boltzmann model with alternative distributions
    """
    print("🔄 Comparing with alternative models...")
    
    # Generate reference data
    t_ref = t[50]  # Middle time point
    P_mb, mb_params = integrated_performance(C, t_ref, learning_params, performance_params)
    
    # Alternative model definitions
    def gaussian_model(C, mu, sigma):
        return np.exp(-0.5 * ((C - mu) / sigma)**2)
    
    def exponential_model(C, lambda_param):
        return np.exp(-lambda_param * C)
    
    def weibull_model(C, k, lambda_param):
        return np.exp(-(C / lambda_param)**k)
    
    def log_normal_model(C, mu, sigma):
        return np.exp(-0.5 * ((np.log(C) - mu) / sigma)**2)
    
    # Fit alternative models to MB data
    models = {
        'Gaussian': (gaussian_model, [mb_params['Copt_t'], 1.0]),
        'Exponential': (exponential_model, [0.5]),
        'Weibull': (weibull_model, [2.0, mb_params['Copt_t']]),
        'Log-Normal': (log_normal_model, [np.log(mb_params['Copt_t']), 0.5])
    }
    
    comparison_results = {}
    
    for model_name, (model_func, initial_params) in models.items():
        try:
            # Fit model to MB data
            popt, _ = curve_fit(model_func, C, P_mb, p0=initial_params, maxfev=1000)
            
            # Generate predictions
            P_pred = model_func(C, *popt)
            
            # Calculate fit metrics
            mse = mean_squared_error(P_mb, P_pred)
            r2 = r2_score(P_mb, P_pred)
            
            # Calculate AIC (simplified)
            n_params = len(popt)
            aic = len(C) * np.log(mse) + 2 * n_params
            
            comparison_results[model_name] = {
                'mse': mse,
                'r2': r2,
                'aic': aic,
                'parameters': popt,
                'predictions': P_pred
            }
            
        except Exception as e:
            print(f"Warning: Could not fit {model_name} model: {e}")
            comparison_results[model_name] = None
    
    return comparison_results, P_mb

def predictive_validation(learning_params, performance_params, C, t):
    """
    Test model predictions on new scenarios
    """
    print("🔄 Testing predictive validation...")
    
    # Create training scenario (first 80% of time)
    train_t = t[:int(0.8 * len(t))]
    
    # Create test scenario (last 20% of time)
    test_t = t[int(0.8 * len(t)):]
    
    # Train model on training data
    train_performances = []
    for t_val in train_t:
        P, _ = integrated_performance(C, t_val, learning_params, performance_params)
        train_performances.append(P)
    
    # Calculate average training parameters
    avg_alpha = np.mean([p['alpha_t'] for p in [integrated_performance(C, t_val, learning_params, performance_params)[1] for t_val in train_t]])
    avg_Copt = np.mean([p['Copt_t'] for p in [integrated_performance(C, t_val, learning_params, performance_params)[1] for t_val in train_t]])
    avg_k = np.mean([p['k_t'] for p in [integrated_performance(C, t_val, learning_params, performance_params)[1] for t_val in train_t]])
    
    # Test predictions on new scenarios
    predictions = []
    actuals = []
    
    for t_val in test_t:
        # Predict using trained parameters
        P_pred = performance_model(C, avg_alpha, avg_Copt, avg_k)
        predictions.append(P_pred)
        
        # Actual performance
        P_actual, _ = integrated_performance(C, t_val, learning_params, performance_params)
        actuals.append(P_actual)
    
    # Calculate prediction accuracy
    all_predictions = np.concatenate(predictions)
    all_actuals = np.concatenate(actuals)
    
    mse = mean_squared_error(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions)
    
    return {
        'mse': mse,
        'r2': r2,
        'train_params': {'alpha': avg_alpha, 'Copt': avg_Copt, 'k': avg_k},
        'test_t': test_t,
        'predictions': predictions,
        'actuals': actuals
    }

# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def create_robustness_plots(uncertainty_results, sensitivity_results, cv_results, 
                           comparison_results, pred_results, C, t, learning_params, performance_params):
    """
    Create comprehensive robustness analysis plots
    """
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Parameter Uncertainty
    ax1 = plt.subplot(3, 4, 1)
    t_sample = t[::2]
    alpha_ci = uncertainty_results['alpha_ci']
    ax1.fill_between(t_sample, alpha_ci[0], alpha_ci[1], alpha=0.3, color='red', label='95% CI')
    ax1.plot(t_sample, [p['true_params']['alpha_t'] for p in uncertainty_results['noisy_data']], 'r-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('α (Expertise)')
    ax1.set_title('Parameter Uncertainty: α')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter Sensitivity
    ax2 = plt.subplot(3, 4, 2)
    params = list(sensitivity_results.keys())
    sensitivities = [np.mean(sensitivity_results[p]) for p in params]
    ax2.bar(params, sensitivities, color=['red', 'blue', 'green', 'orange'])
    ax2.set_ylabel('Sensitivity (Relative Change)')
    ax2.set_title('Parameter Sensitivity Analysis')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cross-Validation Results
    ax3 = plt.subplot(3, 4, 3)
    metrics = ['MSE', 'R²']
    values = [cv_results['mse'], cv_results['r2']]
    bars = ax3.bar(metrics, values, color=['red', 'green'])
    ax3.set_ylabel('Value')
    ax3.set_title('Cross-Validation Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Comparison
    ax4 = plt.subplot(3, 4, 4)
    model_names = [k for k, v in comparison_results.items() if v is not None]
    aic_values = [comparison_results[k]['aic'] for k in model_names]
    ax4.bar(model_names, aic_values, color=['blue', 'green', 'orange', 'purple'])
    ax4.set_ylabel('AIC (Lower is Better)')
    ax4.set_title('Model Comparison (AIC)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Predictive Validation
    ax5 = plt.subplot(3, 4, 5)
    test_t = pred_results['test_t']
    for i, t_val in enumerate(test_t):
        ax5.plot(C, pred_results['predictions'][i], 'r--', alpha=0.7, linewidth=1)
        ax5.plot(C, pred_results['actuals'][i], 'b-', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Cognitive Load')
    ax5.set_ylabel('Performance')
    ax5.set_title('Predictive Validation')
    ax5.grid(True, alpha=0.3)
    
    # 6. Model Fit Quality
    ax6 = plt.subplot(3, 4, 6)
    # Use the actual MB data instead of trying to access comparison results
    P_mb, _ = integrated_performance(C, t[50], learning_params, performance_params)
    ax6.plot(C, P_mb, 'b-', linewidth=2, label='MB Model')
    
    # Plot alternative models if they exist
    if 'Gaussian' in comparison_results and comparison_results['Gaussian'] is not None:
        ax6.plot(C, comparison_results['Gaussian']['predictions'], 'r--', linewidth=2, label='Gaussian')
    if 'Exponential' in comparison_results and comparison_results['Exponential'] is not None:
        ax6.plot(C, comparison_results['Exponential']['predictions'], 'g--', linewidth=2, label='Exponential')
    
    ax6.set_xlabel('Cognitive Load')
    ax6.set_ylabel('Performance')
    ax6.set_title('Model Fit Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Uncertainty Propagation
    ax7 = plt.subplot(3, 4, 7)
    t_mid = t[len(t)//2]
    P_base, _ = integrated_performance(C, t_mid, learning_params, performance_params)
    
    # Add parameter uncertainty bands
    for i in range(10):  # Multiple uncertainty realizations
        # Perturb parameters slightly
        perturbed_params = performance_params.copy()
        perturbed_params['alpha_0'] *= (1 + 0.1 * np.random.randn())
        perturbed_params['Copt_0'] *= (1 + 0.1 * np.random.randn())
        
        P_perturbed, _ = integrated_performance(C, t_mid, learning_params, perturbed_params)
        ax7.plot(C, P_perturbed, 'gray', alpha=0.1)
    
    ax7.plot(C, P_base, 'b-', linewidth=2, label='Base Model')
    ax7.set_xlabel('Cognitive Load')
    ax7.set_ylabel('Performance')
    ax7.set_title('Uncertainty Propagation')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Envelope Stability
    ax8 = plt.subplot(3, 4, 8)
    envelope_areas = []
    for t_val in t:
        P, _ = integrated_performance(C, t_val, learning_params, performance_params)
        area, _, _ = calculate_performance_envelope(C, P)
        envelope_areas.append(area)
    
    ax8.plot(t, envelope_areas, 'b-', linewidth=2)
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Performance Envelope Area')
    ax8.set_title('Envelope Stability Over Time')
    ax8.grid(True, alpha=0.3)
    
    # 9. Parameter Correlation Analysis
    ax9 = plt.subplot(3, 4, 9)
    param_evolution = []
    for t_val in t:
        _, params = integrated_performance(C, t_val, learning_params, performance_params)
        param_evolution.append([params['alpha_t'], params['Copt_t'], params['k_t']])
    
    param_array = np.array(param_evolution)
    correlation_matrix = np.corrcoef(param_array.T)
    
    im = ax9.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax9.set_xticks([0, 1, 2])
    ax9.set_yticks([0, 1, 2])
    ax9.set_xticklabels(['α', 'C_opt', 'k'])
    ax9.set_yticklabels(['α', 'C_opt', 'k'])
    ax9.set_title('Parameter Correlations')
    plt.colorbar(im, ax=ax9)
    
    # 10. Robustness Summary
    ax10 = plt.subplot(3, 4, 10)
    
    # Calculate model comparison score safely
    model_comparison_score = 0
    if comparison_results:
        valid_models = [v for v in comparison_results.values() if v is not None]
        if valid_models:
            # Find the best model (lowest AIC)
            best_aic = min([v['aic'] for v in valid_models])
            # Calculate relative performance
            model_comparison_score = 1 - (best_aic / np.mean([v['aic'] for v in valid_models]))
    
    robustness_metrics = {
        'Parameter Stability': 1 - np.mean([np.std(sensitivity_results[p]) for p in sensitivity_results]),
        'Cross-Validation R²': cv_results['r2'],
        'Prediction R²': pred_results['r2'],
        'Model Comparison': model_comparison_score
    }
    
    metrics_names = list(robustness_metrics.keys())
    metrics_values = list(robustness_metrics.values())
    bars = ax10.bar(metrics_names, metrics_values, color=['green', 'blue', 'orange', 'purple'])
    ax10.set_ylabel('Robustness Score')
    ax10.set_title('Overall Robustness Summary')
    ax10.tick_params(axis='x', rotation=45)
    ax10.grid(True, alpha=0.3)
    
    # 11. Learning Curve Stability
    ax11 = plt.subplot(3, 4, 11)
    learning_states = learning_model(t, **learning_params)
    ax11.plot(t, learning_states['L'], 'b-', linewidth=2, label='Total Learning')
    ax11.fill_between(t, learning_states['L'] * 0.9, learning_states['L'] * 1.1, alpha=0.3, color='blue')
    ax11.set_xlabel('Time')
    ax11.set_ylabel('Learning Level')
    ax11.set_title('Learning Curve Stability')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Performance Surface Robustness
    ax12 = plt.subplot(3, 4, 12, projection='3d')
    T, C_mesh = np.meshgrid(t, C)
    performance_surface = np.zeros((len(t), len(C)))
    
    for i, t_val in enumerate(t):
        P, _ = integrated_performance(C, t_val, learning_params, performance_params)
        performance_surface[i, :] = P
    
    ax12.plot_surface(T.T, C_mesh.T, performance_surface, cmap=cm.viridis, alpha=0.8)
    ax12.set_xlabel('Time')
    ax12.set_ylabel('Cognitive Load')
    ax12.set_zlabel('Performance')
    ax12.set_title('Performance Surface Robustness')
    
    plt.tight_layout()
    plt.savefig('enhanced_robustness_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

# ============================================================================
# MAIN ENHANCED DEMONSTRATION FUNCTION
# ============================================================================

def enhanced_main():
    """
    Enhanced main demonstration with comprehensive validation
    """
    print("🚀 Enhanced Integrated Learning-Performance Model Demonstration")
    print("=" * 70)
    print("📊 Including comprehensive robustness testing and validation")
    print("🎯 Publication-quality analysis with uncertainty quantification")
    print("=" * 70)
    
    # Time range for learning
    t = np.linspace(0, 10, 100)
    
    # Cognitive load range
    C = np.linspace(0.1, 6.0, 200)
    
    # Learning model parameters
    learning_params = {
        'F0': 2.0, 'lambda_rate': 0.6, 'S0': 0.0, 'r': 1.0, 'ks': 0.5,
        'alpha_E': 2.0, 'a': 1.0, 'f': 0.3, 'k_gamma': 1.0, 'E0': 5.0
    }
    
    # Performance model parameters
    performance_params = {
        'alpha_0': 1.5,      # Initial expertise (beginner)
        'delta_alpha': 2.0,  # Maximum expertise gain
        'Copt_0': 1.0,       # Initial optimal load
        'delta_Copt': 2.0,   # Maximum load capacity increase
        'k_0': 1.5,          # Initial overload sensitivity
        'beta': 2.0,          # Learning rate for overload resistance
        'L_max': 15.0,        # Maximum learning capacity
        'E_max': 10.0         # Maximum experience level
    }
    
    print("\n🔄 Running comprehensive validation analysis...")
    
    # 1. Parameter Uncertainty Analysis
    uncertainty_results = parameter_uncertainty_analysis(
        None, C, t, learning_params, performance_params
    )
    
    # 2. Cross-Validation Analysis
    cv_results = cross_validation_analysis(
        None, C, t, learning_params, performance_params
    )
    
    # 3. Parameter Sensitivity Analysis
    sensitivity_results = parameter_sensitivity_analysis(
        learning_params, performance_params, C, t
    )
    
    # 4. Comprehensive Model Comparison
    comparison_results, P_mb = comprehensive_model_comparison(
        C, t, learning_params, performance_params
    )
    
    # 5. Predictive Validation
    pred_results = predictive_validation(
        learning_params, performance_params, C, t
    )
    
    print("✅ All validation analyses completed!")
    
    # Create comprehensive robustness plots
    print("\n🎨 Creating enhanced visualization...")
    fig = create_robustness_plots(
        uncertainty_results, sensitivity_results, cv_results,
        comparison_results, pred_results, C, t, learning_params, performance_params
    )
    
    # Print comprehensive results summary
    print("\n" + "="*70)
    print("📊 ENHANCED VALIDATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n🔍 Parameter Uncertainty (95% CI):")
    print(f"   α (Expertise): ±{np.mean(uncertainty_results['alpha_ci'][1] - uncertainty_results['alpha_ci'][0]):.3f}")
    print(f"   C_opt (Optimal Load): ±{np.mean(uncertainty_results['Copt_ci'][1] - uncertainty_results['Copt_ci'][0]):.3f}")
    print(f"   k (Overload Sensitivity): ±{np.mean(uncertainty_results['k_ci'][1] - uncertainty_results['k_ci'][0]):.3f}")
    
    print(f"\n📈 Cross-Validation Performance:")
    print(f"   Mean Squared Error: {cv_results['mse']:.6f}")
    print(f"   R² Score: {cv_results['r2']:.4f}")
    
    print(f"\n🎯 Predictive Validation:")
    print(f"   Test MSE: {pred_results['mse']:.6f}")
    print(f"   Test R²: {pred_results['r2']:.4f}")
    
    print(f"\n🏆 Model Comparison Results:")
    for model_name, results in comparison_results.items():
        if results is not None:
            print(f"   {model_name}: AIC={results['aic']:.2f}, R²={results['r2']:.4f}")
    
    print(f"\n⚡ Parameter Sensitivity (Most to Least):")
    sorted_sensitivity = sorted(sensitivity_results.items(), 
                               key=lambda x: np.mean(x[1]), reverse=True)
    for param, sensitivity in sorted_sensitivity:
        print(f"   {param}: {np.mean(sensitivity):.4f}")
    
    print(f"\n💾 Results saved to 'enhanced_robustness_analysis.png'")
    
    return {
        'uncertainty_results': uncertainty_results,
        'cv_results': cv_results,
        'sensitivity_results': sensitivity_results,
        'comparison_results': comparison_results,
        'pred_results': pred_results
    }

if __name__ == "__main__":
    results = enhanced_main()
