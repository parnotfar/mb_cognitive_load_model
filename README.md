# Maxwell-Boltzmann Cognitive Load Model

## Overview

This repository contains the implementation and simulation code for the Maxwell-Boltzmann cognitive performance model, as described in the paper "Maxwell--Boltzmann Dynamics in Cognitive Performance: A Mathematical Framework for Skill-Dependent Asymmetric Load Modeling."

## Repository Structure

```
mb_cognitive_load_model/
├── paper/                    # LaTeX source and bibliography
│   ├── par_not_far_cognitive_model.tex
│   └── references.bib
├── figures/                  # Paper figures and visualizations
├── code/                     # Python simulation implementation
│   ├── src/                 # Simulation source code
│   └── requirements.txt     # Python dependencies
├── outputs/                  # Simulation-generated outputs
└── setup.py                 # Installation script
```

## Quick Start

### Prerequisites
- Python 3.8+
- LaTeX distribution (for paper compilation)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mb_cognitive_load_model.git
cd mb_cognitive_load_model

# Install Python dependencies
cd code
pip install -r requirements.txt
```

### Running the Simulation
```bash
# Run enhanced simulation with validation (recommended)
python src/integrated_model_demo_enhanced.py

# Run basic simulation
python src/integrated_model_demo.py
```

## Model Description

The Maxwell-Boltzmann cognitive model describes performance as a function of cognitive load using a three-parameter formulation:

- **α (alpha)**: Skill level parameter (lower = more skilled)
- **C_opt**: Optimal cognitive load
- **k**: Decay rate parameter

The model captures the asymmetric nature of cognitive performance, with rapid ramp-up and gradual decay patterns that vary by skill level.

## Key Features

- **Mathematical Rigor**: Closed-form solutions for performance envelope calculations
- **Computational Efficiency**: Real-time parameter estimation capabilities
- **Validation Framework**: Bootstrap confidence intervals, cross-validation, and model comparison
- **Dynamic Extension**: Time-dependent cognitive load evolution modeling

## Paper Compilation

To compile the LaTeX paper:

```bash
cd paper
pdflatex par_not_far_cognitive_model.tex
bibtex par_not_far_cognitive_model
pdflatex par_not_far_cognitive_model.tex
pdflatex par_not_far_cognitive_model.tex
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bailey2024maxwell,
  title={Maxwell--Boltzmann Dynamics in Cognitive Performance: A Mathematical Framework for Skill-Dependent Asymmetric Load Modeling},
  author={Bailey, Wes},
  journal={Journal of Mathematical Psychology},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

This is an academic research repository. For questions or collaboration, please open an issue or contact the author.

## Acknowledgments

This work was developed as part of research into cognitive performance modeling and its applications in sports psychology and decision-making theory.
