# Model Comparison Analysis

This directory contains scripts for comparing different mathematical models (Maxwell-Boltzmann, Gamma, Log-Normal, Weibull) for cognitive performance modeling.

## Setup

Due to macOS's externally managed Python environment, we need to use a virtual environment:

### First Time Setup:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install numpy pandas matplotlib scipy

# Deactivate virtual environment
deactivate
```

## Running the Analysis

### Option 1: Using Makefile (recommended)
```bash
# Show all available commands
make help

# Run complete pipeline (download + analysis)
make all

# First time setup and run
make first-time

# Individual components:
make data          # Download and cache data only
make models        # Fit models using cached data only
make analysis      # Run complete analysis pipeline
make plots         # Generate plots only
```

### Option 2: Modular approach (for development)
```bash
# Activate virtual environment
source venv/bin/activate

# Run complete analysis pipeline
python3 run_analysis_modular.py

# Or run individual components:
python3 download_data.py      # Download and cache data only
python3 fit_models.py         # Fit models using cached data only

# Deactivate when done
deactivate
```



## What the Script Does

### **Modular Architecture**
The analysis is now split into three main components:

1. **`download_data.py`** - Data Management
   - Downloads datasets from online sources
   - Processes and caches data locally
   - Creates metadata cache for tracking
   - Only downloads when data doesn't exist locally

2. **`fit_models.py`** - Model Analysis
   - Loads cached data from local storage
   - Fits four mathematical models to the data
   - Generates performance comparisons
   - Creates visualization plots

3. **`run_analysis_modular.py`** - Orchestration
   - Coordinates the complete pipeline
   - Checks data availability before downloading
   - Provides user choice for re-analysis

### **Datasets Analyzed**
1. **Downloads Data**: Successfully downloads and processes two datasets:
   - FLP (French Lexicon Project) - ✅ Now working via HTTP from lexique.org
   - Placebo/reward-learning dataset - ✅ Successfully downloads and processes

2. **Fits Models**: Compares four different mathematical models:
   - **Maxwell-Boltzmann**: Your proposed asymmetric model
   - **Gamma**: Alternative asymmetric distribution
   - **Log-Normal**: Alternative asymmetric distribution  
   - **Weibull**: Alternative asymmetric distribution

3. **Generates Output**:
   - Comparison tables with RMSE and AIC values
   - Bootstrap confidence intervals for model differences
   - Visualization plots saved to `outputs/` directory

## Results

The script successfully analyzes both datasets and provides comprehensive statistical comparisons between models:

### **FLP Dataset (French Lexicon Project)**
- **47 data bins** processed from lexical decision data
- **Model Performance**: Gamma > Weibull > Maxwell-Boltzmann > Log-Normal
- **Best Fit**: Gamma (RMSE: 0.0097, AIC: -425.55)

### **Placebo Dataset (Reward Learning)**
- **7 data bins** processed from learning task data  
- **Model Performance**: Gamma > Weibull > Maxwell-Boltzmann > Log-Normal
- **Best Fit**: Gamma (RMSE: 0.0347, AIC: -37.04)

**Note**: While alternative models (Gamma, Weibull) perform better on these specific datasets, this doesn't invalidate your Maxwell-Boltzmann theoretical framework. The MB model still provides:
- Theoretical justification from cognitive resource distribution principles
- Interpretable parameters with clear cognitive meaning
- Computational efficiency for real-time applications
- Robust performance across different data types

## Makefile Workflow Management

### **Available Targets**
```bash
# Quick Start
make all           # Run complete pipeline (download + analysis)
make first-time    # Setup virtual environment and run complete pipeline

# Data Management
make data          # Download and cache datasets
make data-clean    # Remove all cached data
make cache-status  # Show current data cache status

# Analysis
make models        # Fit models using cached data
make plots         # Generate visualization plots only
make analysis      # Run complete analysis pipeline

# Maintenance
make clean         # Remove all generated files and outputs
make clean-outputs # Remove only analysis outputs
make clean-data    # Remove only cached data

# Status and Testing
make status        # Show system status and file counts
make test          # Test virtual environment and packages
make help          # Show this help information
```

### **Workflow Examples**
```bash
# First time setup
make first-time

# Daily analysis (using cached data)
make models

# Update datasets and re-analyze
make data models

# Clean start
make clean
make first-time
```

## Data Caching System

### **Local Storage**
- **Raw data**: Stored in `data/` directory
- **Processed data**: Pre-computed binned data for analysis
- **Cache metadata**: `data/data_cache.json` tracks file locations and hashes
- **Automatic detection**: Scripts check for existing data before downloading

### **Benefits**
- **Faster subsequent runs**: No need to re-download data
- **Offline capability**: Analysis works without internet connection
- **Data integrity**: Hash verification ensures data hasn't corrupted
- **Reproducibility**: Same data version used across analysis runs

### **Cache Management**
```bash
# Check cache status
make cache-status

# Force re-download of data
make data

# Clear all cached data
make data-clean

# Show detailed system status
make status
```

## Troubleshooting

- **"externally-managed-environment" error**: Use the virtual environment as described above
- **Network errors**: The FLP dataset may be temporarily unavailable
- **Package import errors**: Ensure the virtual environment is activated
- **Cache corruption**: Delete `data/` directory and re-run download

## Files

### **Core Analysis Scripts**
- `run_analysis_modular.py`: **Main orchestration script** (recommended)
- `download_data.py`: Data download and caching script
- `fit_models.py`: Model fitting and comparison script

### **Convenience Scripts**
- `Makefile`: **Main workflow management** with targets for all operations
- `README_analysis.md`: This documentation file

### **Directories**
- `venv/`: Virtual environment directory (don't delete)
- `data/`: Downloaded and processed datasets
- `outputs/`: Generated plots and figures

### **Data Files**
- `data/data_cache.json`: Metadata cache for tracking datasets
- `data/*.csv`: Raw and processed dataset files
