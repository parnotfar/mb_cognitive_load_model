#!/usr/bin/env python3
"""
Data Download Script for Model Comparison Analysis
Downloads and caches datasets locally to avoid repeated downloads
"""

import os
import pandas as pd
from urllib.request import urlopen
import hashlib
import json
from datetime import datetime
import numpy as np # Added missing import for numpy

# Configuration
DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "data_cache.json")

# Dataset definitions
DATASETS = {
    "flp": {
        "name": "French Lexicon Project",
        "url": "http://www.lexique.org/databases/FrenchLexiconProject/FLP.words.csv",
        "filename": "FLP_words.csv",
        "description": "Lexical decision data for 38,840 French words",
        "columns": ["item", "ntrials", "err", "rt", "sd", "rtz", "nused", 
                   "cfreqmovies", "lcfreqmovies", "cfreqbooks", "lcfreqbooks", 
                   "nletters", "nsyllables"]
    },
    "placebo": {
        "name": "Placebo/Reward Learning",
        "url": "https://raw.githubusercontent.com/ihrke/2016-placebo-tdcs-study/master/data/export/placebo_tdcs_learn.csv",
        "filename": "placebo_learning.csv",
        "description": "Learning task performance under different difficulty levels",
        "columns": ["Participant", "pair", "condition", "ACC", "RT", "reward"]
    }
}

def ensure_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✓ Created data directory: {DATA_DIR}")

def get_file_hash(filepath):
    """Calculate SHA256 hash of a file"""
    if not os.path.exists(filepath):
        return None
    
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def download_dataset(dataset_id, dataset_info, force_download=False):
    """Download a dataset and save it locally"""
    filepath = os.path.join(DATA_DIR, dataset_info["filename"])
    
    # Check if file exists and is valid
    if os.path.exists(filepath) and not force_download:
        print(f"✓ {dataset_info['name']} already exists locally: {filepath}")
        return filepath
    
    print(f"Downloading {dataset_info['name']}...")
    print(f"URL: {dataset_info['url']}")
    
    try:
        # Download data
        response = urlopen(dataset_info['url'], timeout=30)
        data = response.read()
        response.close()
        
        # Save to file
        with open(filepath, 'wb') as f:
            f.write(data)
        
        print(f"✓ Downloaded successfully: {filepath}")
        print(f"  File size: {len(data):,} bytes")
        
        return filepath
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None

def process_flp_data(filepath):
    """Process FLP data and create processed version"""
    print("Processing FLP data...")
    
    try:
        # Read raw data
        flp = pd.read_csv(filepath)
        print(f"  Raw data: {len(flp)} rows, {len(flp.columns)} columns")
        
        # Map columns to accuracy and frequency
        flp["acc"] = 1 - flp["err"]  # Convert error rate to accuracy
        flp["freq"] = flp["cfreqmovies"]  # Use movie frequency
        
        # Build load proxy: higher when frequency is lower
        flp["C"] = -np.log(flp["freq"].clip(lower=1e-6))
        flp["C"] = (flp["C"] - flp["C"].mean()) / flp["C"].std()
        flp["P"] = flp["acc"].clip(0,1)
        
        # Bin data for analysis
        def bin_curve(df, nbins=50):
            q = pd.qcut(df["C"], nbins, duplicates="drop")
            g = df.groupby(q, observed=True).agg(C=("C","mean"), P=("P","mean"), n=("P","size")).dropna()
            return g.reset_index(drop=True)
        
        flp_binned = bin_curve(flp, nbins=50)
        
        # Save processed data
        processed_filepath = os.path.join(DATA_DIR, "FLP_processed.csv")
        flp_binned.to_csv(processed_filepath, index=False)
        
        print(f"  Processed data: {len(flp_binned)} bins")
        print(f"  Saved to: {processed_filepath}")
        
        return processed_filepath
        
    except Exception as e:
        print(f"✗ FLP processing failed: {e}")
        return None

def process_placebo_data(filepath):
    """Process placebo data and create processed version"""
    print("Processing placebo data...")
    
    try:
        # Read raw data
        rl = pd.read_csv(filepath)
        print(f"  Raw data: {len(rl)} rows, {len(rl.columns)} columns")
        
        # Map difficulty to load; add condition uncertainty as shift
        diff_map = {1:0.2, 2:0.3, 3:0.4}   # 80/20, 70/30, 60/40
        cond_shift = {"N":0.0, "A":0.05, "B":0.10}  # baseline, low-uncertainty, high-uncertainty
        
        rl = rl.rename(columns={"pair":"pair","condition":"cond","RT":"RT","ACC":"ACC"})
        rl["C_raw"] = rl["pair"].map(diff_map).fillna(0)
        rl["C_raw"] += rl["cond"].map(cond_shift).fillna(0)
        
        # Subject-level accuracy by C
        tmp = rl.groupby(["Participant","C_raw"], observed=True)["ACC"].mean().reset_index()
        # Normalize C
        tmp["C"] = (tmp["C_raw"] - tmp["C_raw"].mean())/tmp["C_raw"].std()
        tmp["P"] = tmp["ACC"].clip(0,1)
        
        # Bin data for analysis
        def bin_curve(df, nbins=12):
            q = pd.qcut(df["C"], nbins, duplicates="drop")
            g = df.groupby(q, observed=True).agg(C=("C","mean"), P=("P","mean"), n=("P","size")).dropna()
            return g.reset_index(drop=True)
        
        rl_binned = bin_curve(tmp, nbins=12)
        
        # Save processed data
        processed_filepath = os.path.join(DATA_DIR, "placebo_processed.csv")
        rl_binned.to_csv(processed_filepath, index=False)
        
        print(f"  Processed data: {len(rl_binned)} bins")
        print(f"  Saved to: {processed_filepath}")
        
        return processed_filepath
        
    except Exception as e:
        print(f"✗ Placebo processing failed: {e}")
        return None

def update_cache(dataset_id, filepath, processed_filepath):
    """Update the data cache with file information"""
    if not os.path.exists(CACHE_FILE):
        cache = {}
    else:
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
        except:
            cache = {}
    
    cache[dataset_id] = {
        "raw_file": filepath,
        "processed_file": processed_filepath,
        "raw_hash": get_file_hash(filepath),
        "processed_hash": get_file_hash(processed_filepath),
        "last_updated": datetime.now().isoformat(),
        "dataset_info": DATASETS[dataset_id]
    }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"✓ Updated cache: {CACHE_FILE}")

def main():
    """Main function to download and process all datasets"""
    print("=" * 60)
    print("DATA DOWNLOAD AND PROCESSING SCRIPT")
    print("=" * 60)
    
    # Ensure data directory exists
    ensure_data_directory()
    
    results = {}
    
    # Download and process each dataset
    for dataset_id, dataset_info in DATASETS.items():
        print(f"\n{'='*40}")
        print(f"Processing: {dataset_info['name']}")
        print(f"{'='*40}")
        
        # Download raw data
        raw_filepath = download_dataset(dataset_id, dataset_info)
        if not raw_filepath:
            print(f"✗ Skipping {dataset_id} due to download failure")
            continue
        
        # Process data
        if dataset_id == "flp":
            processed_filepath = process_flp_data(raw_filepath)
        elif dataset_id == "placebo":
            processed_filepath = process_placebo_data(raw_filepath)
        else:
            print(f"Unknown dataset type: {dataset_id}")
            continue
        
        if processed_filepath:
            # Update cache
            update_cache(dataset_id, raw_filepath, processed_filepath)
            results[dataset_id] = {
                "raw": raw_filepath,
                "processed": processed_filepath,
                "status": "success"
            }
        else:
            results[dataset_id] = {"status": "processing_failed"}
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    for dataset_id, result in results.items():
        if result["status"] == "success":
            print(f"✅ {DATASETS[dataset_id]['name']}: Ready for analysis")
        else:
            print(f"❌ {DATASETS[dataset_id]['name']}: {result['status']}")
    
    print(f"\nData files stored in: {DATA_DIR}")
    print(f"Cache file: {CACHE_FILE}")
    print(f"\nYou can now run the model fitting script!")

if __name__ == "__main__":
    main()
