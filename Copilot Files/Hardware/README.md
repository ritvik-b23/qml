# Hardware Experiments - IBM Torino Simulation

This folder contains quantum intrusion detection experiments running on **IBM Torino fake backend** with realistic hardware constraints.

## Overview

While the main experiments use idealized quantum simulators (`default.qubit`), these hardware experiments simulate real IBM Torino quantum hardware with:
- **Realistic noise models** (gate errors, readout errors, decoherence)
- **Hardware topology** (limited qubit connectivity, SWAP gates)
- **Shot-based execution** (1024 shots, statistical sampling)
- **Transpilation overhead** (circuit optimization for hardware gates)

## Files

### Core Pipeline
- **`utils/hardware_quantum_pipeline.py`** - Hardware-adapted quantum IDS pipeline
  - Extends base pipeline with IBM Torino backend integration
  - Implements shot-based quantum kernel and QNN execution
  - Handles noise model and transpilation
  - Configurable model selection (QSVM, QNN, QKNN)

### Experiment Notebooks
- **`01_hardware_train_test_network.ipynb`** - Train/Test Network dataset on hardware
  - Uses 13 qubits (same as simulator for comparison)
  - Two configurations: fast (QSVM+QNN) and full (all models)
  - Expected runtime: 2-6 hours

- **`02_hardware_cicids2017.ipynb`** - CICIDS2017 dataset on hardware
  - Uses 10 qubits with 20,000 sampled rows
  - Memory-efficient loading with chunked processing
  - Expected runtime: 2-6 hours

- **`03_hardware_analysis.ipynb`** - **Conference-ready hardware analysis**
  - IBM Torino specifications (T1/T2, gate errors, topology)
  - Performance degradation quantification
  - Publication-quality LaTeX tables and 600 DPI figures
  - **Most important for conference paper**

### Results
- **`results/`** - Hardware experiment outputs
  - CSV files with metrics for each dataset
  - LaTeX tables for publication
  - Consolidated results across experiments

## Quick Start

### 1. Run Hardware Experiments

```python
# Open 01_hardware_train_test_network.ipynb
# Choose configuration:
config = config_fast  # QSVM + QNN only (~2-3 hours)
# or
config = config_full  # All models (~4-6 hours)

# Run experiment
results_df = run_dataset_experiment_hardware(...)
```

### 2. Analyze Hardware Characteristics

```python
# Open 03_hardware_analysis.ipynb
# Run all cells to generate:
# - Hardware specifications table
# - Topology visualization
# - Noise analysis plots
# - Performance comparison
# - LaTeX tables for paper
```

## Configuration Options

### HardwareConfig Parameters

```python
HardwareConfig(
    backend_name="fake_torino",  # IBM Torino backend
    shots=1024,                   # Fixed at 1024
    num_features=10,              # 10 or 13 qubits
    optimization_level=2,         # Transpilation optimization

    # Model selection flags
    run_qnn_models=True,         # Include QNN/VQC
    run_qknn=False,              # Skip QKNN for faster run

    # Sample sizes (same as simulator)
    qsvm_train_size=300,         # Per batch (10 batches total)
    qsvm_test_size=100,          # Per batch
    qnn_train_size=3000,
    qnn_test_size=1000,
)
```

### Configuration Presets

**Option 1: Fast** (QSVM + QNN only)
- Models: QSVM, QNN_Base, QNN_Reupload, VQC_AltAnsatz
- Runtime: ~2-3 hours per dataset
- Use case: Quick comparison, testing

**Option 2: Full** (All models)
- Models: QSVM, QNN×3, QKNN
- Runtime: ~4-6 hours per dataset
- Use case: Comprehensive analysis, publication

## Model Details

### QSVM (Quantum Support Vector Machine)
- **Batched execution**: 10 batches × 300 train / 100 test
- **Total samples**: 3000 train, 1000 test (same as other models)
- **Hardware impact**: Kernel values affected by shot noise
- **Robust to noise**: Kernel-based methods show resilience

### QNN/VQC (Quantum Neural Networks)
- **Shot-based training**: Expectation values from 1024 shots
- **Noisy gradients**: Hardware noise affects backpropagation
- **Circuit depth**: Moderate depth circuits (~50-100 gates)
- **More sensitive**: Parametrized circuits affected by gate errors

### QKNN (Quantum K-Nearest Neighbors)
- **Query-based**: Each test sample queries training set
- **Slowest model**: Most time-consuming on hardware
- **Distance computation**: Quantum kernel similarity
- **Skip option**: Can be disabled for faster runs

## Hardware Backend: IBM Torino

**IBM Torino (FakeTorinoV2) Specifications:**
- **Number of qubits**: 133
- **Basis gates**: RZ, SX, X, ECR (native IBM gates)
- **Coupling map**: Heavy-hex lattice topology
- **Average T1**: ~100-200 μs (relaxation time)
- **Average T2**: ~80-150 μs (dephasing time)
- **1Q gate error**: ~1×10⁻⁴ (0.01%)
- **2Q gate error**: ~1×10⁻² (1%)
- **Readout error**: ~1-2%

## Expected Results

### Performance Degradation
Hardware results typically show **5-15% degradation** compared to simulator:
- **Accuracy**: -5 to -15%
- **F1 Score**: -5 to -15%
- **AUPRC**: -5 to -15%

### Key Observations
1. **QSVM** shows good robustness to hardware noise
2. **QNN/VQC** more sensitive to gate errors
3. **Classical models** unaffected (provide upper bound)
4. **Shot noise** introduces variance but results are stable

## Outputs

### Experiment Results
- **CSV files**: `results/*_metrics_summary.csv`
- **Plots**: ROC/PR curves in `Plots PDF/` and `Images/`
- **Consolidated**: `results/all_datasets_metrics_summary.csv`

### Hardware Analysis Outputs
- **LaTeX tables**: `results/*.tex` (ready for \include{})
- **PDF figures**: `Plots PDF/*.pdf` (vector graphics)
- **High-res images**: `Images/*.png` and `*.jpg` (600 DPI)

### Conference-Ready Materials
From `03_hardware_analysis.ipynb`:
1. `ibm_torino_specs_table.tex` - Hardware specifications
2. `performance_comparison_table.tex` - Simulator vs. hardware
3. `ibm_torino_topology.pdf` - Coupling map visualization
4. `ibm_torino_noise_analysis.pdf` - Error distributions
5. `simulator_vs_hardware_comparison.pdf` - Performance bars

## Dependencies

Required packages:
```
qiskit >= 1.0
qiskit-ibm-runtime >= 0.18
qiskit-aer >= 0.13
pennylane >= 0.35
pennylane-qiskit >= 0.35
torch >= 2.0
scikit-learn >= 1.3
pandas >= 2.0
matplotlib >= 3.7
seaborn >= 0.12
networkx >= 3.0
```

Install with:
```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer pennylane pennylane-qiskit torch scikit-learn pandas matplotlib seaborn networkx
```

## Troubleshooting

### Import Errors
If you see "Hardware dependencies not available":
```bash
pip install qiskit-ibm-runtime qiskit-aer pennylane-qiskit
```

### Memory Issues
If CICIDS2017 crashes:
- Chunked loading is already implemented
- Reduce `cicids_sample_n` if needed (default: 20000)

### Long Runtime
To speed up experiments:
- Use `config_fast` (QSVM + QNN only)
- Set `run_qknn=False`
- Reduce sample sizes for testing

### Backend Not Found
If FakeTorinoV2 is not available:
```python
from qiskit_ibm_runtime.fake_provider import FakeTorino
# or
from qiskit_aer import AerSimulator
backend = AerSimulator()  # Fallback to generic simulator
```

## Comparison with Simulator

### Simulator Experiments (Main Folder)
- Location: `/home/ritvik/Python/Copilot Files/`
- Backend: `default.qubit` (PennyLane, idealized)
- Execution: Analytic, noise-free
- Runtime: Faster (~1-2 hours)

### Hardware Experiments (This Folder)
- Location: `/home/ritvik/Python/Copilot Files/Hardware/`
- Backend: `FakeTorinoV2` (IBM Torino simulation)
- Execution: Shot-based, realistic noise
- Runtime: Slower (~2-6 hours)

**Use both** for complete analysis:
1. Run simulator experiments first (main folder)
2. Run hardware experiments second (this folder)
3. Analyze performance gap (`03_hardware_analysis.ipynb`)

## Citation

If you use this hardware simulation in your research, please cite:
- IBM Quantum: https://quantum.ibm.com/
- Qiskit: https://qiskit.org/
- PennyLane: https://pennylane.ai/

## Contact

For questions about hardware experiments:
- Check `03_hardware_analysis.ipynb` for detailed explanations
- Review `utils/hardware_quantum_pipeline.py` for implementation details
- Refer to main `README.md` in parent folder for general project info

## Notes

- **Non-invasive**: Does not modify any files in parent `Copilot Files/` folder
- **Independent**: Can run without affecting simulator experiments
- **Reproducible**: Fixed seed (42) ensures consistent results
- **Publication-ready**: All outputs formatted for conference/journal papers

---

**Last Updated**: March 2026
**IBM Torino Backend**: FakeTorinoV2 (133 qubits)
