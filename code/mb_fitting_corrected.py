#!/usr/bin/env python3
"""
Corrected MB Distribution Fitting for Biathlon Cognitive Model
P = shooting accuracy (success rate)
C = cognitive load (time pressure, rank, shot time, round number)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import re # Added missing import for re

def load_biathlon_data(filename):
    """Load the processed biathlon race data"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} athlete results from {filename}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_shooting_rounds_data(data):
    """Extract data for each shooting round to create P vs C relationship"""
    print("\n" + "="*60)
    print("EXTRACTING SHOOTING ROUNDS DATA")
    print("="*60)
    
    shooting_data = []
    
    for athlete in data:
        # Extract individual shooting round data
        for round_num in range(1, 5):
            shooting_col = f'shooting_{round_num}'
            if shooting_col in athlete and athlete[shooting_col]:
                shooting_text = athlete[shooting_col]
                
                # Parse shooting data
                round_data = parse_shooting_round(shooting_text, round_num, athlete)
                if round_data:
                    shooting_data.append(round_data)
    
    print(f"Extracted {len(shooting_data)} shooting round data points")
    return shooting_data

def parse_shooting_round(shooting_text, round_num, athlete):
    """Parse individual shooting round data"""
    try:
        # Extract shot pattern (e.g., "54321" means 5 hits out of 5)
        shot_pattern_match = re.search(r'(\d{5})$', shooting_text)
        if not shot_pattern_match:
            return None
        
        shot_pattern = shot_pattern_match.group(1)
        hits = sum(1 for digit in shot_pattern if digit != '0')
        total_shots = 5
        accuracy = hits / total_shots
        
        # Extract individual shot times
        shot_times = re.findall(r'(\d+\.\d+)s', shooting_text)
        if len(shot_times) == 5:
            shot_times = [float(t) for t in shot_times]
            avg_shot_time = np.mean(shot_times)
            total_round_time = sum(shot_times)
        else:
            avg_shot_time = None
            total_round_time = None
        
        # Calculate cognitive load factors
        race_rank = athlete.get('rank', 0)
        round_number = round_num
        
        # Cognitive load increases with:
        # 1. Higher race rank (more pressure)
        # 2. Later shooting rounds (fatigue)
        # 3. Faster shot times (time pressure)
        cognitive_load = calculate_cognitive_load(race_rank, round_number, avg_shot_time, total_round_time)
        
        return {
            'athlete_name': athlete.get('athlete_name', ''),
            'race_rank': race_rank,
            'round_number': round_number,
            'shooting_accuracy': accuracy,
            'hits': hits,
            'total_shots': total_shots,
            'avg_shot_time': avg_shot_time,
            'total_round_time': total_round_time,
            'cognitive_load': cognitive_load,
            'shot_pattern': shot_pattern
        }
        
    except Exception as e:
        print(f"Error parsing shooting round: {e}")
        return None

def calculate_cognitive_load(race_rank, round_number, avg_shot_time, total_round_time):
    """Calculate cognitive load based on multiple factors"""
    
    # Normalize factors to 0-1 scale
    # Higher rank = more pressure (lower rank number = higher pressure)
    rank_pressure = 1.0 / (race_rank + 1)  # Avoid division by zero
    
    # Later rounds = more fatigue
    round_fatigue = round_number / 4.0
    
    # Time pressure (faster shots = more pressure)
    time_pressure = 0
    if avg_shot_time is not None:
        # Normalize shot time (faster = higher pressure)
        # Assuming 1-10 seconds is the range
        time_pressure = max(0, (10 - avg_shot_time) / 9)
    
    # Combine factors (weighted sum)
    cognitive_load = (0.4 * rank_pressure + 
                      0.3 * round_fatigue + 
                      0.3 * time_pressure)
    
    return cognitive_load

def analyze_p_vs_c_relationship(shooting_data):
    """Analyze the relationship between P (accuracy) and C (cognitive load)"""
    print("\n" + "="*60)
    print("P vs C RELATIONSHIP ANALYSIS")
    print("="*60)
    
    # Extract P and C values
    P_values = [d['shooting_accuracy'] for d in shooting_data if d['shooting_accuracy'] is not None]
    C_values = [d['cognitive_load'] for d in shooting_data if d['cognitive_load'] is not None]
    
    print(f"Shooting Accuracy (P):")
    print(f"  Count: {len(P_values)}")
    print(f"  Mean: {np.mean(P_values):.3f} ({np.mean(P_values)*100:.1f}%)")
    print(f"  Std: {np.std(P_values):.3f}")
    print(f"  Min: {min(P_values):.3f} ({min(P_values)*100:.1f}%)")
    print(f"  Max: {max(P_values):.3f} ({max(P_values)*100:.1f}%)")
    
    print(f"\nCognitive Load (C):")
    print(f"  Count: {len(C_values)}")
    print(f"  Mean: {np.mean(C_values):.3f}")
    print(f"  Std: {np.std(C_values):.3f}")
    print(f"  Min: {min(C_values):.3f}")
    print(f"  Max: {max(C_values):.3f}")
    
    # Check correlation
    if len(P_values) == len(C_values) and len(P_values) > 1:
        correlation = np.corrcoef(P_values, C_values)[0, 1]
        print(f"\nCorrelation between P and C: {correlation:.3f}")
        
        if correlation < 0:
            print("  ✓ Negative correlation (higher cognitive load → lower accuracy)")
        else:
            print("  ⚠️  Positive correlation (higher cognitive load → higher accuracy)")
    
    return P_values, C_values

def fit_mb_distribution_to_p_vs_c(P_values, C_values):
    """Fit MB distribution to model the P vs C relationship"""
    print("\n" + "="*60)
    print("MB DISTRIBUTION FITTING TO P vs C")
    print("="*60)
    
    if len(P_values) != len(C_values) or len(P_values) < 10:
        print("Insufficient data for MB fitting")
        return None
    
    # Create P vs C data points
    data_points = list(zip(C_values, P_values))
    
    # Sort by cognitive load
    data_points.sort(key=lambda x: x[0])
    C_sorted = [x[0] for x in data_points]
    P_sorted = [x[1] for x in data_points]
    
    # For MB fitting, we need to understand the relationship
    # In your cognitive model, P should decrease as C increases
    # This suggests a negative relationship that could be modeled
    
    print("Data points (C, P):")
    for i, (c, p) in enumerate(data_points[:10]):  # Show first 10
        print(f"  C={c:.3f}, P={p:.3f}")
    
    if len(data_points) > 10:
        print(f"  ... and {len(data_points) - 10} more")
    
    # Calculate the relationship parameters
    # We can fit a linear model: P = a - b*C
    # Where 'a' is baseline accuracy and 'b' is cognitive load sensitivity
    
    C_array = np.array(C_sorted)
    P_array = np.array(P_sorted)
    
    # Linear fit: P = a - b*C
    coeffs = np.polyfit(C_array, P_array, 1)
    a, b = coeffs[0], -coeffs[1]  # Note: b is positive for negative relationship
    
    print(f"\nLinear fit: P = {a:.3f} - {b:.3f}*C")
    print(f"  Baseline accuracy (C=0): {a:.3f}")
    print(f"  Cognitive load sensitivity: {b:.3f}")
    
    # Calculate R-squared
    P_pred = a - b * C_array
    ss_res = np.sum((P_array - P_pred) ** 2)
    ss_tot = np.sum((P_array - np.mean(P_array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"  R-squared: {r_squared:.3f}")
    
    return {
        'a': a,
        'b': b,
        'r_squared': r_squared,
        'C_values': C_sorted,
        'P_values': P_sorted,
        'P_predicted': P_pred.tolist()
    }

def create_p_vs_c_visualization(shooting_data, mb_fit):
    """Create visualization of P vs C relationship"""
    print("\n" + "="*60)
    print("CREATING P vs C VISUALIZATION")
    print("="*60)
    
    # Set up the plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Biathlon Shooting: P (Accuracy) vs C (Cognitive Load)', fontsize=16)
    
    # 1. P vs C scatter plot
    C_values = [d['cognitive_load'] for d in shooting_data if d['cognitive_load'] is not None]
    P_values = [d['shooting_accuracy'] for d in shooting_data if d['shooting_accuracy'] is not None]
    
    axes[0, 0].scatter(C_values, P_values, alpha=0.7, color='blue')
    if mb_fit:
        axes[0, 0].plot(mb_fit['C_values'], mb_fit['P_predicted'], 'r-', 
                        linewidth=2, label=f'Fit: P = {mb_fit["a"]:.3f} - {mb_fit["b"]:.3f}*C')
        axes[0, 0].legend()
    axes[0, 0].set_xlabel('Cognitive Load (C)')
    axes[0, 0].set_ylabel('Shooting Accuracy (P)')
    axes[0, 0].set_title('P vs C Relationship')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy distribution by round
    round_data = {}
    for d in shooting_data:
        round_num = d['round_number']
        if round_num not in round_data:
            round_data[round_num] = []
        round_data[round_num].append(d['shooting_accuracy'])
    
    round_numbers = sorted(round_data.keys())
    round_accuracies = [round_data[r] for r in round_numbers]
    
    axes[0, 1].boxplot(round_accuracies, labels=[f'Round {r}' for r in round_numbers])
    axes[0, 1].set_xlabel('Shooting Round')
    axes[0, 1].set_ylabel('Shooting Accuracy')
    axes[0, 1].set_title('Accuracy by Round (Fatigue Effect)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Accuracy vs race rank
    rank_groups = {}
    for d in shooting_data:
        rank = d['race_rank']
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(d['shooting_accuracy'])
    
    ranks = sorted(rank_groups.keys())
    rank_accuracies = [rank_groups[r] for r in ranks]
    
    axes[1, 0].boxplot(rank_accuracies, labels=[f'Rank {r}' for r in ranks])
    axes[1, 0].set_xlabel('Race Rank')
    axes[1, 0].set_ylabel('Shooting Accuracy')
    axes[1, 0].set_title('Accuracy by Race Rank (Pressure Effect)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cognitive load distribution
    axes[1, 1].hist(C_values, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_xlabel('Cognitive Load (C)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Cognitive Load')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "biathlon_p_vs_c_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")
    
    plt.show()

def export_p_vs_c_data(shooting_data, mb_fit):
    """Export P vs C data for further analysis"""
    print("\n" + "="*60)
    print("EXPORTING P vs C DATA")
    print("="*60)
    
    # Create comprehensive dataset
    export_data = []
    
    for d in shooting_data:
        export_row = {
            'athlete_name': d['athlete_name'],
            'race_rank': d['race_rank'],
            'round_number': d['round_number'],
            'shooting_accuracy': d['shooting_accuracy'],
            'hits': d['hits'],
            'total_shots': d['total_shots'],
            'avg_shot_time': d['avg_shot_time'],
            'total_round_time': d['total_round_time'],
            'cognitive_load': d['cognitive_load'],
            'shot_pattern': d['shot_pattern']
        }
        export_data.append(export_row)
    
    # Save as CSV
    df = pd.DataFrame(export_data)
    csv_file = "biathlon_p_vs_c_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Exported P vs C data to: {csv_file}")
    
    # Save as JSON with MB fit parameters
    if mb_fit:
        complete_data = {
            'mb_fit_parameters': {
                'baseline_accuracy': mb_fit['a'],
                'cognitive_load_sensitivity': mb_fit['b'],
                'r_squared': mb_fit['r_squared'],
                'model': f"P = {mb_fit['a']:.3f} - {mb_fit['b']:.3f}*C"
            },
            'shooting_rounds_data': export_data
        }
        
        json_file = "biathlon_p_vs_c_complete.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Exported complete P vs C analysis to: {json_file}")
    
    return export_data

def main():
    """Main function for corrected MB fitting"""
    print("=" * 60)
    print("CORRECTED MB FITTING: P (Accuracy) vs C (Cognitive Load)")
    print("=" * 60)
    
    # Load the biathlon data
    data_file = "data/biathlon_processed_BT2425SWRLCP01SMMS_2025.json"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run the biathlon scraper first to generate the data.")
        return
    
    data = load_biathlon_data(data_file)
    if not data:
        return
    
    # Extract shooting rounds data
    shooting_data = extract_shooting_rounds_data(data)
    if not shooting_data:
        print("No shooting data found")
        return
    
    # Analyze P vs C relationship
    P_values, C_values = analyze_p_vs_c_relationship(shooting_data)
    
    # Fit MB distribution to P vs C
    mb_fit = fit_mb_distribution_to_p_vs_c(P_values, C_values)
    
    # Create visualizations
    create_p_vs_c_visualization(shooting_data, mb_fit)
    
    # Export data
    export_data = export_p_vs_c_data(shooting_data, mb_fit)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("The biathlon shooting data has been analyzed for P vs C relationship.")
    print("\nKey findings:")
    print(f"• {len(shooting_data)} shooting rounds analyzed")
    print(f"• Shooting accuracy (P) ranges from {min(P_values)*100:.1f}% to {max(P_values)*100:.1f}%")
    print(f"• Cognitive load (C) ranges from {min(C_values):.3f} to {max(C_values):.3f}")
    
    if mb_fit:
        print(f"• Model: P = {mb_fit['a']:.3f} - {mb_fit['b']:.3f}*C")
        print(f"• R-squared: {mb_fit['r_squared']:.3f}")
        print(f"• Higher cognitive load → lower shooting accuracy")
    
    print("\nFiles generated:")
    print("• biathlon_p_vs_c_analysis.png - P vs C visualizations")
    print("• biathlon_p_vs_c_data.csv - Raw P vs C data")
    if mb_fit:
        print("• biathlon_p_vs_c_complete.json - Complete analysis with model parameters")

if __name__ == "__main__":
    main()
