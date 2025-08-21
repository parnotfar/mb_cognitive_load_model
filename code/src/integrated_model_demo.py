#!/usr/bin/env python3
"""
Integrated Learning-Performance Model Demonstration
Shows how learning affects cognitive performance parameters over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

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
    Maxwell-Boltzmann performance model - FIXED VERSION
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

def llm_integration_example():
    """
    Demonstrate how the model integrates with LLM decision making
    """
    print("\n🤖 LLM Integration Example:")
    print("=" * 50)
    
    # Simulate a player's current state
    current_time = 3.5  # 3.5 time units into learning
    current_cognitive_load = 2.0  # Current task complexity
    
    # Get current performance parameters
    learning_params = {
        'F0': 2.0, 'lambda_rate': 0.6, 'S0': 0.0, 'r': 1.0, 'ks': 0.5,
        'alpha_E': 2.0, 'a': 1.0, 'f': 0.3, 'k_gamma': 1.0, 'E0': 5.0
    }
    
    performance_params = {
        'alpha_0': 1.5, 'delta_alpha': 2.0, 'Copt_0': 1.0, 'delta_Copt': 2.0,
        'k_0': 1.5, 'beta': 2.0, 'L_max': 15.0, 'E_max': 10.0
    }
    
    # Calculate current state
    learning_state = learning_model(current_time, **learning_params)
    C = np.linspace(0.1, 6.0, 100)
    P, params = integrated_performance(C, current_time, learning_params, performance_params)
    
    # LLM decision making
    current_performance = np.interp(current_cognitive_load, C, P)
    optimal_load = params['Copt_t']
    cognitive_capacity = 1.0 - abs(current_cognitive_load - optimal_load) / optimal_load
    
    print(f"📊 Current Player State:")
    print(f"   Learning Progress: {learning_state['L']:.2f}/{performance_params['L_max']:.1f}")
    print(f"   Current Performance: {current_performance:.3f}")
    print(f"   Optimal Load: {optimal_load:.2f}")
    print(f"   Current Load: {current_cognitive_load:.2f}")
    print(f"   Cognitive Capacity: {cognitive_capacity:.3f}")
    
    print(f"\n🎯 LLM Response Strategy:")
    
    if cognitive_capacity < 0.3:
        print("   🟡 LOW CAPACITY: Generate simple, clear advice")
        print("      Example: 'Hit it straight. Focus on the target.'")
    elif cognitive_capacity < 0.7:
        print("   🟢 MEDIUM CAPACITY: Provide standard technical advice")
        print("      Example: 'Aim for the center of the green. Check your stance.'")
    else:
        print("   🔵 HIGH CAPACITY: Offer complex strategic guidance")
        print("      Example: 'Consider wind direction, green slope, and pin position for optimal approach.'")
    
    # Advice timing optimization
    if learning_state['L'] < performance_params['L_max'] * 0.3:
        print(f"\n📚 LEARNING STAGE: Early - Focus on fundamentals")
    elif learning_state['L'] < performance_params['L_max'] * 0.7:
        print(f"\n📚 LEARNING STAGE: Intermediate - Build on basics")
    else:
        print(f"\n📚 LEARNING STAGE: Advanced - Fine-tune technique")
    
    return {
        'current_performance': current_performance,
        'cognitive_capacity': cognitive_capacity,
        'optimal_load': optimal_load,
        'learning_progress': learning_state['L'] / performance_params['L_max']
    }

def main():
    """
    Main demonstration function
    """
    print("Integrated Learning-Performance Model Demonstration")
    print("=" * 60)
    
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
    
    # Calculate learning progression
    learning_states = learning_model(t, **learning_params)
    
    # Create performance surface over time
    performance_surface = np.zeros((len(t), len(C)))
    envelope_evolution = []
    parameter_evolution = []
    
    for i, t_val in enumerate(t):
        P, params = integrated_performance(C, t_val, learning_params, performance_params)
        performance_surface[i, :] = P
        
        # Calculate performance envelope
        area, C_min, C_max = calculate_performance_envelope(C, P)
        envelope_evolution.append({
            't': t_val,
            'area': area,
            'C_min': C_min,
            'C_max': C_max
        })
        
        parameter_evolution.append({
            't': t_val,
            'alpha': params['alpha_t'],
            'Copt': params['Copt_t'],
            'k': params['k_t'],
            'learning': params['learning']['L']
        })
    
    # Convert to DataFrames for analysis
    envelope_df = pd.DataFrame(envelope_evolution)
    param_df = pd.DataFrame(parameter_evolution)
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Learning progression
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t, learning_states['F'], 'r-', label='Failure Rate', linewidth=2)
    ax1.plot(t, learning_states['S'], 'g-', label='Success Rate', linewidth=2)
    ax1.plot(t, learning_states['E'], 'b-', label='Experience', linewidth=2)
    ax1.plot(t, learning_states['L'], 'k-', label='Total Learning', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Rate/Level')
    ax1.set_title('Learning Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter evolution
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t, param_df['alpha'], 'r-', label='α (Expertise)', linewidth=2)
    ax2.plot(t, param_df['Copt'], 'g-', label='C_opt (Optimal Load)', linewidth=2)
    ax2.plot(t, param_df['k'], 'b-', label='k (Overload Sensitivity)', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Performance Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance envelope evolution
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t, envelope_df['area'], 'k-', linewidth=2)
    ax3.fill_between(t, envelope_df['C_min'], envelope_df['C_max'], alpha=0.3, color='blue')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Performance Envelope')
    ax3.set_title('Performance Envelope Evolution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance surface (3D)
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    T, C_mesh = np.meshgrid(t, C)
    ax4.plot_surface(T.T, C_mesh.T, performance_surface, cmap=cm.viridis, alpha=0.8)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cognitive Load')
    ax4.set_zlabel('Performance')
    ax4.set_title('Performance Surface Over Time')
    
    # 5. Performance curves at different time points
    ax5 = plt.subplot(2, 3, 5)
    time_points = [0, 2, 5, 8, 10]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, t_val in enumerate(time_points):
        t_idx = np.argmin(np.abs(t - t_val))
        P, _ = integrated_performance(C, t_val, learning_params, performance_params)
        ax5.plot(C, P, color=colors[i], linewidth=2, label=f't={t_val}')
    
    ax5.set_xlabel('Cognitive Load')
    ax5.set_ylabel('Performance')
    ax5.set_title('Performance Curves at Different Times')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training optimization
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate optimal training load
    delta = 0.5  # Training load increment
    C_train = param_df['Copt'] + delta * np.sqrt(param_df['learning'] / performance_params['L_max'])
    
    ax6.plot(t, param_df['Copt'], 'g-', label='Optimal Load', linewidth=2)
    ax6.plot(t, C_train, 'r--', label='Training Load', linewidth=2)
    ax6.fill_between(t, param_df['Copt'], C_train, alpha=0.3, color='orange')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Cognitive Load')
    ax6.set_title('Training Load Optimization')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integrated_model_demo.png', dpi=150, bbox_inches='tight')
    
    # Print summary statistics
    print("\nModel Summary:")
    print(f"Initial Performance Envelope: {envelope_df['area'].iloc[0]:.3f}")
    print(f"Final Performance Envelope: {envelope_df['area'].iloc[-1]:.3f}")
    print(f"Envelope Growth: {envelope_df['area'].iloc[-1] / envelope_df['area'].iloc[0]:.1f}x")
    
    print(f"\nParameter Evolution:")
    print(f"α: {param_df['alpha'].iloc[0]:.2f} → {param_df['alpha'].iloc[-1]:.2f}")
    print(f"C_opt: {param_df['Copt'].iloc[0]:.2f} → {param_df['Copt'].iloc[-1]:.2f}")
    print(f"k: {param_df['k'].iloc[0]:.2f} → {param_df['k'].iloc[-1]:.2f}")
    
    print(f"\nLearning Progress:")
    print(f"Total Learning: {param_df['learning'].iloc[-1]:.2f}")
    print(f"Experience: {param_df['learning'].iloc[-1]:.2f}")
    
    # Show LLM integration example
    llm_state = llm_integration_example()
    
    # Don't show plot interactively - just save it
    plt.close()
    
    return {
        'envelope_df': envelope_df,
        'param_df': param_df,
        'performance_surface': performance_surface,
        'learning_states': learning_states,
        'llm_state': llm_state
    }

if __name__ == "__main__":
    results = main()
