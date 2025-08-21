#!/usr/bin/env python3
"""
Model Fitting Script for Cognitive Performance Analysis
Uses locally cached data to fit and compare different mathematical models
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "data_cache.json")
OUTPUT_DIR = "outputs"

def load_cached_data():
    """Load data from cache and verify files exist"""
    if not os.path.exists(CACHE_FILE):
        print("❌ No data cache found. Please run download_data.py first.")
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    except Exception as e:
        print(f"❌ Error reading cache file: {e}")
        return None
    
    available_datasets = {}
    
    for dataset_id, info in cache.items():
        processed_file = info.get("processed_file")
        if processed_file and os.path.exists(processed_file):
            try:
                data = pd.read_csv(processed_file)
                available_datasets[dataset_id] = {
                    "data": data,
                    "info": info,
                    "name": info["dataset_info"]["name"]
                }
                print(f"✅ Loaded {info['dataset_info']['name']}: {len(data)} data points")
            except Exception as e:
                print(f"❌ Error loading {dataset_id}: {e}")
        else:
            print(f"❌ Processed data not found for {dataset_id}")
    
    return available_datasets

def mb_shape(C, alpha, Copt):
    """
    Robust Maxwell-Boltzmann shape function that handles edge cases
    """
    # Ensure Copt is positive and not too close to zero
    Copt = max(Copt, 0.1)
    
    # Shift C so that Copt becomes the center
    C_shifted = C - Copt
    
    # Calculate k based on alpha and Copt
    k = alpha / Copt
    
    # Use safer power and exponential operations
    try:
        # For positive C_shifted, use the original formula
        pos_mask = C_shifted >= 0
        y = np.zeros_like(C)
        
        if np.any(pos_mask):
            y[pos_mask] = (C[pos_mask] / Copt)**alpha * np.exp(-k * C_shifted[pos_mask])
        
        # For negative C_shifted, use a modified approach to avoid complex numbers
        neg_mask = C_shifted < 0
        if np.any(neg_mask):
            # Use absolute value for negative shifts, but maintain the shape
            y[neg_mask] = (np.abs(C[neg_mask]) / Copt)**alpha * np.exp(-k * np.abs(C_shifted[neg_mask]))
        
        # Normalize to peak at 1
        y_max = np.max(y)
        if y_max > 0:
            y = y / y_max
        else:
            y = np.ones_like(C)
            
        return y
        
    except Exception as e:
        print(f"Error in mb_shape: {e}")
        return np.ones_like(C)

def gamma_shape(C, k, theta, mu):
    X = np.clip(C - mu, 1e-9, None)
    y = (X**(k-1)) * np.exp(-X/theta)
    return y / np.max(y)

def lognorm_shape(C, m, s, mu):
    X = np.clip(C - mu, 1e-9, None)
    y = np.exp(-((np.log(X)-m)**2)/(2*s**2))
    return y / np.max(y)

def weibull_shape(C, beta, lam, mu):
    X = np.clip(C - mu, 1e-9, None)
    y = (X**(beta-1)) * np.exp(- (X/lam)**beta)
    return y / np.max(y)

def bounded(model_y, Pmin, Pmax):
    return Pmin + (Pmax - Pmin) * model_y

def fit_model(C, P, shape_name):
    """Fit a model to the data"""
    C = np.asarray(C); P = np.asarray(P)
    Pmin0 = max(0.0, P.min()*0.8)
    Pmax0 = min(1.0, P.max()*1.05)

    if shape_name == "MB":
        f = lambda C, alpha, Copt, Pmin, Pmax: bounded(mb_shape(C, alpha, Copt), Pmin, Pmax)
        p0 = [1.5, 0.0, Pmin0, Pmax0]  # More conservative initial values
        # Set bounds that are compatible with initial values and ensure Pmin < Pmax
        bounds = ([0.5, -0.5, Pmin0*0.9, Pmax0*0.9], [3.0, 0.5, Pmin0*1.1, Pmax0*1.1])
    elif shape_name == "Gamma":
        f = lambda C, k, theta, mu, Pmin, Pmax: bounded(gamma_shape(C, k, theta, mu), Pmin, Pmax)
        p0 = [2.0, 1.0, 0.0, Pmin0, Pmax0]  # Center mu at 0
        bounds = ([0.5, 1e-3, -0.5, Pmin0*0.9, Pmax0*0.9], [10.0, 10.0, 0.5, Pmin0*1.1, Pmax0*1.1])
    elif shape_name == "LogNormal":
        f = lambda C, m, s, mu, Pmin, Pmax: bounded(lognorm_shape(C, m, s, mu), Pmin, Pmax)
        p0 = [0.0, 0.6, 0.0, Pmin0, Pmax0]  # Center mu at 0
        bounds = ([-5.0, 0.1, -0.5, Pmin0*0.9, Pmax0*0.9], [5.0, 5.0, 0.5, Pmin0*1.1, Pmax0*1.1])
    elif shape_name == "Weibull":
        f = lambda C, beta, lam, mu, Pmin, Pmax: bounded(weibull_shape(C, beta, lam, mu), Pmin, Pmax)
        p0 = [2.0, 1.5, 0.0, Pmin0, Pmax0]  # Center mu at 0
        bounds = ([0.5, 1e-3, -0.5, Pmin0*0.9, Pmax0*0.9], [10.0, 10.0, 0.5, Pmin0*1.1, Pmax0*1.1])
    else:
        raise ValueError("Unknown shape")

    try:
        popt, pcov = curve_fit(f, C, P, p0=p0, bounds=bounds, maxfev=100000)
        P_hat = f(C, *popt)
        resid = P - P_hat
        rmse = float(np.sqrt(np.mean(resid**2)))
        k_params = len(popt)
        n = len(C)
        sse = float(np.sum(resid**2))
        aic = n*np.log(sse/n) + 2*k_params
        return {"name": shape_name, "params": popt, "rmse": rmse, "aic": aic, "predict": lambda x: f(x, *popt)}
    except Exception as e:
        print(f"Warning: Fitting failed for {shape_name}: {e}")
        return {"name": shape_name, "params": None, "rmse": float('inf'), "aic": float('inf'), "predict": None}

def compare_models(dataset_id, dataset_info, data):
    """Compare different models on a dataset"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {dataset_info['name']}")
    print(f"{'='*60}")
    
    C, P = data["C"].values, data["P"].values
    
    print(f"Data summary:")
    print(f"  Cognitive Load range: {C.min():.3f} to {C.max():.3f}")
    print(f"  Performance range: {P.min():.3f} to {P.max():.3f}")
    print(f"  Number of data points: {len(C)}")
    
    # Fit all models
    models = ["MB", "Gamma", "LogNormal", "Weibull"]
    results = []
    
    for model_name in models:
        result = fit_model(C, P, model_name)
        if result["params"] is not None:
            results.append(result)
            print(f"  ✓ {model_name} fitted successfully")
        else:
            print(f"  ✗ {model_name} fitting failed")
    
    if not results:
        print("❌ No models fitted successfully")
        return
    
    # Create comparison table
    results.sort(key=lambda x: x["aic"])
    tbl = pd.DataFrame([{k:v for k,v in r.items() if k in ("name","rmse","aic")} for r in results])
    
    print(f"\nModel Comparison Results:")
    print(tbl.to_string(index=False))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(C, P, s=30, alpha=0.7, label="Data", color='black')
    
    # Plot fitted curves
    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (result, color, ls) in enumerate(zip(results, colors, linestyles)):
        if result["predict"] is not None:
            C_smooth = np.linspace(C.min(), C.max(), 200)
            P_smooth = result["predict"](C_smooth)
            plt.plot(C_smooth, P_smooth, color=color, linestyle=ls, 
                    linewidth=2, label=f"{result['name']} (RMSE: {result['rmse']:.4f})")
    
    plt.xlabel("Cognitive Load (standardized)")
    plt.ylabel("Performance")
    plt.title(f"{dataset_info['name']}: Model Fits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    safe_name = dataset_info['name'].replace(' ', '_').replace('/', '_').replace('—', '_').replace('(', '').replace(')', '')
    plot_file = os.path.join(OUTPUT_DIR, f"{safe_name}_fits.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved: {plot_file}")
    
    # Bootstrap confidence intervals for AIC differences
    if len(results) > 1:
        print(f"\nBootstrap Confidence Intervals (AIC differences):")
        base_model = results[0]  # Best model
        
        if base_model["name"] != "MB":
            print(f"  Note: {base_model['name']} performed best, but comparing to Maxwell-Boltzmann")
        
        # Find MB model
        mb_result = next((r for r in results if r["name"] == "MB"), None)
        if mb_result and mb_result["params"] is not None:
            rng = np.random.default_rng(0)
            
            def boot_stat(data, idx):
                Cb = C[idx]; Pb = P[idx]
                try:
                    base = fit_model(Cb, Pb, "MB")["aic"]
                    others = [fit_model(Cb, Pb, nm)["aic"] for nm in ["Gamma","LogNormal","Weibull"] if nm != "MB"]
                    return np.array([o - base for o in others])
                except:
                    return np.array([float('inf')] * 3)
            
            idxs = np.arange(len(C))
            boots = []
            for _ in range(300):  # Light bootstrap
                samp = rng.integers(0, len(C), len(C))
                boots.append(boot_stat((C,P), samp))
            
            boots = np.vstack(boots)
            ci = np.quantile(boots, [0.025, 0.5, 0.975], axis=0)
            
            other_models = ["Gamma", "LogNormal", "Weibull"]
            ci_tbl = pd.DataFrame(ci.T, columns=["AICΔ 2.5%","AICΔ 50%","AICΔ 97.5%"], index=other_models)
            print(ci_tbl.to_string())
    
    return results

def main():
    """Main function to run model fitting on cached data"""
    print("=" * 60)
    print("MODEL FITTING AND COMPARISON SCRIPT")
    print("=" * 60)
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ Created output directory: {OUTPUT_DIR}")
    
    # Load cached data
    datasets = load_cached_data()
    if not datasets:
        print("\n❌ No datasets available. Please run download_data.py first.")
        return
    
    print(f"\n✅ Loaded {len(datasets)} datasets for analysis")
    
    # Analyze each dataset
    all_results = {}
    for dataset_id, dataset_info in datasets.items():
        results = compare_models(dataset_id, dataset_info, dataset_info["data"])
        if results:
            all_results[dataset_id] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for dataset_id, results in all_results.items():
        if results:
            best_model = results[0]
            print(f"✅ {datasets[dataset_id]['name']}: {best_model['name']} performed best (RMSE: {best_model['rmse']:.4f})")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
