#!/usr/bin/env python3
"""
Main Analysis Orchestration Script
Coordinates data download and model fitting processes
"""

import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {script_name}: {e}")
        return False

def check_data_status():
    """Check if data is already available"""
    data_dir = "data"
    cache_file = os.path.join(data_dir, "data_cache.json")
    
    if os.path.exists(cache_file):
        try:
            import json
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            
            available_datasets = []
            for dataset_id, info in cache.items():
                processed_file = info.get("processed_file")
                if processed_file and os.path.exists(processed_file):
                    available_datasets.append(dataset_id)
            
            if available_datasets:
                print(f"✅ Found cached data for {len(available_datasets)} datasets:")
                for dataset_id in available_datasets:
                    dataset_name = cache[dataset_id]["dataset_info"]["name"]
                    print(f"   - {dataset_name}")
                return True
            else:
                print("⚠️  Cache file exists but no processed data found")
                return False
        except Exception as e:
            print(f"⚠️  Error reading cache: {e}")
            return False
    else:
        print("❌ No data cache found")
        return False

def main():
    """Main orchestration function"""
    print("=" * 60)
    print("COGNITIVE PERFORMANCE MODEL ANALYSIS")
    print("=" * 60)
    print("This script coordinates the complete analysis pipeline:")
    print("1. Data download and caching")
    print("2. Model fitting and comparison")
    print("=" * 60)
    
    # Check if data is already available
    data_available = check_data_status()
    
    if data_available:
        print(f"\n💡 Data is already available. You can:")
        print(f"   - Run model fitting only: python3 fit_models.py")
        print(f"   - Re-download data: python3 download_data.py")
        print(f"   - Continue with full analysis")
        
        response = input(f"\nContinue with full analysis? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Analysis cancelled.")
            return
    
    # Step 1: Download and process data
    print(f"\n📥 STEP 1: Downloading and processing data...")
    if not run_script("download_data.py", "Data Download and Processing"):
        print("❌ Data download failed. Cannot proceed with analysis.")
        return
    
    # Brief pause to ensure files are written
    time.sleep(1)
    
    # Step 2: Fit models and generate comparisons
    print(f"\n🔬 STEP 2: Fitting models and generating comparisons...")
    if not run_script("fit_models.py", "Model Fitting and Comparison"):
        print("❌ Model fitting failed.")
        return
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print("What was accomplished:")
    print("✅ Data downloaded and cached locally")
    print("✅ Models fitted to both datasets")
    print("✅ Performance comparisons generated")
    print("✅ Visualization plots created")
    print("✅ Statistical analysis completed")
    
    print(f"\n📁 Files created:")
    print(f"   - Raw data: data/")
    print(f"   - Processed data: data/*_processed.csv")
    print(f"   - Analysis plots: outputs/")
    print(f"   - Data cache: data/data_cache.json")
    
    print(f"\n🚀 Next time you can run just the model fitting:")
    print(f"   python3 fit_models.py")
    
    print(f"\n📊 Or re-download data if needed:")
    print(f"   python3 download_data.py")

if __name__ == "__main__":
    main()
