.# QMLForge

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **QMLForge** – A Modular, GPU-Optimized Quantum Machine Learning Framework for Accurate, Scalable Computational Chemistry on Consumer Hardware

## Overview

QMLForge bridges *ab initio* quantum mechanics with modern equivariant machine learning interatomic potentials (MLIPs) for hybrid simulations. It enables:

- **QM Calculations** (DFT/HF/MP2/CCSD(T)) on systems up to ~100 atoms
- **MLIP Training** with active learning and uncertainty quantification
- **Molecular Dynamics** with GPU acceleration (10-100+ ns/day on RTX 4060)
- **Hybrid QM/ML/MM** multi-scale simulations
- **Inverse Design** via differentiable optimization

**Target Hardware:** Single NVIDIA RTX 4060 (8GB VRAM) + 24GB RAM

## Key Features

- **Quantum Chemistry Backend**: GPU-accelerated PySCF (via GPU4PySCF)
- **Equivariant MLIPs**: MACE/Allegro-style models in PyTorch
- **Differentiable MD**: PyTorch-native simulation engine
- **Active Learning**: Automated data acquisition with uncertainty quantification
- **Property Prediction**: Spectra, free energies, reactivity descriptors
- **Web UI**: Gradio-based interface for non-expert users

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/qmlforge/qmlforge.git
cd qmlforge

# Create conda environment
conda env create -f environment.yml
conda activate qmlforge

# Or install with pip
pip install -e .

# For GPU support
pip install -e ".[gpu]"
```

### Basic Usage

```python
from qmlforge.qm import QMRunner
from ase import Atoms

# Define molecule
water = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])

# Run DFT calculation
qm = QMRunner(method='B3LYP', basis='def2-svp')
energy = qm.calculate_energy(water)
forces = qm.calculate_forces(water)
```

### CLI

```bash
# Run QM calculation
qmlforge run-qm --molecule water.xyz --method B3LYP --basis def2-svp

# Train MLIP
qmlforge train --config configs/training.yaml

# Run MD simulation
qmlforge simulate --model model.pt --system system.xyz --steps 100000
```

## Project Structure

```
qmlforge/
├── qmlforge/
│   ├── qm/           # Quantum chemistry backend
│   ├── data/         # Dataset management
│   ├── models/       # MLIP architectures
│   ├── sim/          # Simulation engine
│   ├── analysis/     # Property prediction
│   ├── cli/          # Command-line interface
│   └── utils/        # Utilities
├── tests/            # Test suite
├── docs/             # Documentation
├── examples/         # Example workflows
└── scripts/          # Utility scripts
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [API Reference](docs/api.md)
- [Examples](examples/)

## Performance Benchmarks

On NVIDIA RTX 4060 (8GB VRAM):

| Task | System Size | Performance |
|------|-------------|-------------|
| DFT (B3LYP/def2-SVP) | 50 atoms | ~5-10 min/geometry |
| MLIP MD | 100 atoms | 10-100 ns/day |
| MLIP Training | <100 atoms | Hours-days/iteration |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

QMLForge is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use QMLForge in your research, please cite:

```bibtex
@software{qmlforge,
  title = {QMLForge: A GPU-Optimized Quantum Machine Learning Framework},
  author = {QMLForge Team},
  year = {2026},
  url = {https://github.com/qmlforge/qmlforge}
}
```

## Acknowledgments

QMLForge builds upon many excellent open-source projects:
- [PySCF](https://pyscf.org/) / [GPU4PySCF](https://github.com/pyscf/gpu4pyscf)
- [PyTorch](https://pytorch.org/)
- [MACE](https://github.com/ACEsuit/mace)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [RDKit](https://www.rdkit.org/)
