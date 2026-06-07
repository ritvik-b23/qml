"""
Hardware Quantum IDS Pipeline
Uses IBM Sherbrooke Fake Backend for realistic hardware simulation

This module extends the base quantum_ids_pipeline to work with IBM hardware backends,
including realistic noise models, gate sets, and connectivity constraints.
"""

import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import base components from parent pipeline
sys.path.append('/home/ritvik/Python/Copilot Files/utils')
from quantum_ids_pipeline import (
    ExperimentConfig,
    ProgressReporter,
    DataBundle,
    ModelOutput,
    evaluate_predictions,
    compute_tpr_at_low_fpr,
    load_train_test_network,
    load_cicids,
    prepare_data,
    run_classical_baselines,
    _plot_roc_pr,
    save_plot_pair,
    set_publication_style,
    ensure_dirs,
    run_fusion_schemes,
)

# Hardware-specific imports
try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from qiskit_aer import AerSimulator
    import pennylane as qml
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from joblib import Parallel, delayed

    HARDWARE_AVAILABLE = True
except ImportError as e:
    HARDWARE_AVAILABLE = False
    IMPORT_ERROR = str(e)


@dataclass
class HardwareConfig(ExperimentConfig):
    """Configuration for hardware-based experiments"""
    backend_name: str = "fake_sherbrooke"
    shots: int = 1024
    optimization_level: int = 2
    coupling_map_aware: bool = True
    noise_model_enabled: bool = True
    run_qnn_models: bool = True  # If False, skip QNN/VQC
    run_qknn: bool = True  # If False, skip QKNN

    # Override output paths for hardware results
    output_root: str = "/home/ritvik/Python/Copilot Files/Hardware/results"


def _get_hardware_device(num_qubits: int, config: HardwareConfig):
    """
    Initialize hardware backend and create PennyLane device

    Returns:
        qml device, fake_backend
    """
    if not HARDWARE_AVAILABLE:
        raise ImportError(f"Hardware dependencies not available: {IMPORT_ERROR}")

    # Initialize IBM Sherbrooke fake backend
    fake_backend = FakeSherbrooke()

    # Create AerSimulator with noise model from fake backend
    if config.noise_model_enabled:
        simulator = AerSimulator.from_backend(fake_backend)
    else:
        # Use AerSimulator without noise for testing
        simulator = AerSimulator()

    # Create PennyLane device with Qiskit backend
    dev = qml.device(
        "qiskit.aer",
        wires=num_qubits,
        backend=simulator,
        shots=config.shots
    )

    return dev, fake_backend


def _quantum_kernel_predict_hardware(
    qml, dev, x1: np.ndarray, x2: np.ndarray, num_features: int
) -> float:
    """
    Compute quantum kernel value between two data points on hardware

    Uses shot-based execution, so results include shot noise
    """
    @qml.qnode(dev)
    def kernel_circuit(a, b):
        qml.AngleEmbedding(a, wires=range(num_features))
        qml.adjoint(qml.AngleEmbedding)(b, wires=range(num_features))
        return qml.probs(wires=range(num_features))

    # Execute circuit - returns shot-based probabilities
    probs = kernel_circuit(x1, x2)
    return float(probs[0])


def _quantum_kernel_matrix_hardware(
    qml,
    dev,
    X1: np.ndarray,
    X2: np.ndarray,
    num_features: int,
    max_workers: int,
    reporter: Optional[ProgressReporter] = None,
    desc: str = "Kernel rows",
) -> np.ndarray:
    """
    Build quantum kernel matrix using hardware backend

    Each kernel value is computed with finite shots, introducing measurement noise
    """
    n1, n2 = len(X1), len(X2)

    def one_row(i: int) -> np.ndarray:
        row = np.zeros(n2)
        for j in range(n2):
            row[j] = _quantum_kernel_predict_hardware(qml, dev, X1[i], X2[j], num_features)
        return row

    row_iter = range(n1)
    if reporter is not None:
        row_iter = reporter.loop(row_iter, desc=desc, total=n1)

    # Use sequential execution (max_workers=1) due to hardware backend constraints
    rows = Parallel(n_jobs=1)(delayed(one_row)(i) for i in row_iter)
    return np.vstack(rows)


def run_qsvm_hardware(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: HardwareConfig,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[List[dict], Dict[str, ModelOutput]]:
    """
    Run QSVM on hardware backend with batched execution

    Processes 10 batches of 300 train / 100 test each (total 3000/1000)
    """
    if not HARDWARE_AVAILABLE:
        return [{"Model": "QSVM", "Status": f"SKIPPED: {IMPORT_ERROR}"}], {}

    rows: List[dict] = []
    outputs: Dict[str, ModelOutput] = {}

    # Batching configuration
    n_batches = 10
    batch_train_size = config.qsvm_train_size  # 300
    batch_test_size = config.qsvm_test_size    # 100
    total_train = n_batches * batch_train_size  # 3000
    total_test = n_batches * batch_test_size    # 1000

    n_train = min(total_train, len(X_train))
    n_test = min(total_test, len(X_test))

    X_all_train = X_train[:n_train]
    y_all_train = y_train[:n_train]
    X_all_test = X_test[:n_test]
    y_all_test = y_test[:n_test]

    # Initialize hardware device
    dev, _ = _get_hardware_device(config.num_features, config)

    # Collect predictions from all batches
    all_hard = []
    all_margin = []
    all_prob = []
    best_c_values = []

    if reporter is not None:
        reporter.step(f"QSVM Hardware: processing {n_batches} batches on {config.backend_name} ({config.shots} shots)")

    batch_iter = range(n_batches)
    if reporter is not None:
        batch_iter = reporter.loop(batch_iter, desc="QSVM HW batches", total=n_batches)

    for batch_idx in batch_iter:
        # Get batch data
        train_start = batch_idx * batch_train_size
        train_end = train_start + batch_train_size
        test_start = batch_idx * batch_test_size
        test_end = test_start + batch_test_size

        X_tr = X_all_train[train_start:train_end]
        y_tr = y_all_train[train_start:train_end]
        X_te = X_all_test[test_start:test_end]

        # Build kernel matrices for this batch
        K_train = _quantum_kernel_matrix_hardware(
            qml, dev, X_tr, X_tr, config.num_features,
            max_workers=1, reporter=None, desc=f"Batch {batch_idx+1} train"
        )

        K_test = _quantum_kernel_matrix_hardware(
            qml, dev, X_te, X_tr, config.num_features,
            max_workers=1, reporter=None, desc=f"Batch {batch_idx+1} test"
        )

        # Train SVC with grid search
        svc = SVC(kernel="precomputed", probability=True)
        grid = GridSearchCV(
            svc, {"C": list(config.qsvm_c_grid)},
            cv=3, scoring="average_precision", n_jobs=1
        )
        grid.fit(K_train, y_tr)
        best = grid.best_estimator_
        best_c_values.append(grid.best_params_['C'])

        # Get predictions
        batch_prob = best.predict_proba(K_test)[:, 1]
        batch_hard = best.predict(K_test)
        batch_margin = best.decision_function(K_test)

        all_prob.extend(batch_prob)
        all_hard.extend(batch_hard)
        all_margin.extend(batch_margin)

    # Convert to arrays
    prob = np.array(all_prob)
    hard = np.array(all_hard)
    margin = np.array(all_margin)

    outputs["QSVM"] = ModelOutput(hard=hard, margin=margin, prob=prob)

    # Evaluate on all test samples
    row = evaluate_predictions(y_all_test, hard, prob, config.target_fprs)

    # Report the most common C value
    most_common_c = Counter(best_c_values).most_common(1)[0][0]
    row.update({
        "Model": "QSVM",
        "Backend": config.backend_name,
        "Shots": config.shots,
        "BestParams": f"C={most_common_c} (mode across {n_batches} batches)",
        "EvalSamples": n_test
    })
    rows.append(row)

    if reporter is not None:
        reporter.step(f"QSVM Hardware: completed {n_batches} batches on {config.backend_name}")

    return rows, outputs


def _train_torch_binary_qnn_hardware(
    qml, torch, nn, optim, DataLoader, TensorDataset,
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    config: HardwareConfig, template: str, layers: int, epochs: int,
    reporter: Optional[ProgressReporter] = None,
) -> ModelOutput:
    """
    Train QNN on hardware backend with shot-based execution
    """
    n_qubits = config.num_features
    dev, _ = _get_hardware_device(n_qubits, config)

    def circuit(inputs, weights):
        if template == "strongly_entangling":
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif template == "reupload":
            for l in range(layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                for q in range(n_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
        else:  # vqc_alt
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qml.BasicEntanglerLayers(weights[:, :, 0], wires=range(n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    qnode = qml.QNode(circuit, dev, interface="torch")

    class HybridQNN(nn.Module):
        def __init__(self):
            super().__init__()
            if template == "vqc_alt":
                weight_shapes = {"weights": (layers, n_qubits, 1)}
            else:
                weight_shapes = {"weights": (layers, n_qubits, 3)}
            self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
            self.fc = nn.Linear(n_qubits, 1)

        def forward(self, x):
            x = self.qlayer(x)
            return self.fc(x).squeeze(1)

    model = HybridQNN()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=config.qnn_batch_size,
        shuffle=True
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.qnn_lr)

    model.train()
    epoch_iter = range(epochs)
    if reporter is not None:
        epoch_iter = reporter.loop(epoch_iter, desc=f"{template} HW epochs", total=epochs)

    for _ in epoch_iter:
        running_loss = 0.0
        batch_count = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1
        if reporter is not None and hasattr(epoch_iter, "set_postfix") and batch_count > 0:
            epoch_iter.set_postfix(loss=f"{running_loss / batch_count:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte).cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        hard = (prob >= 0.5).astype(int)
        margin = logits

    return ModelOutput(hard=hard, margin=margin, prob=prob)


def run_quantum_nn_family_hardware(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    config: HardwareConfig,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[List[dict], Dict[str, ModelOutput]]:
    """
    Run QNN models on hardware backend
    """
    if not HARDWARE_AVAILABLE:
        return [{"Model": "QNN/VQC", "Status": f"SKIPPED: {IMPORT_ERROR}"}], {}

    if not config.run_qnn_models:
        if reporter is not None:
            reporter.step("QNN models skipped per configuration")
        return [], {}

    _, torch, nn, optim, DataLoader, TensorDataset = None, __import__('torch'), __import__('torch.nn'), __import__('torch.optim'), __import__('torch.utils.data').utils.data.DataLoader, __import__('torch.utils.data').utils.data.TensorDataset

    rows: List[dict] = []
    outputs: Dict[str, ModelOutput] = {}

    n_train = min(config.qnn_train_size, len(X_train))
    n_test = min(config.qnn_test_size, len(X_test))
    X_tr = X_train[:n_train]
    y_tr = y_train[:n_train]
    X_te = X_test[:n_test]
    y_te = y_test[:n_test]

    # Base QNN
    if reporter is not None:
        reporter.step(f"QNN Hardware: training on {config.backend_name} with {config.shots} shots")
    qnn_out = _train_torch_binary_qnn_hardware(
        qml, torch, nn, optim, DataLoader, TensorDataset,
        X_tr, y_tr, X_te, config,
        template="strongly_entangling",
        layers=config.qnn_layers,
        epochs=config.qnn_epochs,
        reporter=reporter
    )
    outputs["QNN_Base"] = qnn_out
    row = evaluate_predictions(y_te, qnn_out.hard, qnn_out.prob, config.target_fprs)
    row.update({"Model": "QNN_Base", "Backend": config.backend_name, "Shots": config.shots, "EvalSamples": n_test})
    rows.append(row)

    # Data re-uploading QNN
    if reporter is not None:
        reporter.step("QNN Re-uploading Hardware: training started")
    reup_out = _train_torch_binary_qnn_hardware(
        qml, torch, nn, optim, DataLoader, TensorDataset,
        X_tr, y_tr, X_te, config,
        template="reupload",
        layers=config.qnn_layers,
        epochs=max(20, config.qnn_epochs - 5),
        reporter=reporter
    )
    outputs["QNN_Reupload"] = reup_out
    row = evaluate_predictions(y_te, reup_out.hard, reup_out.prob, config.target_fprs)
    row.update({"Model": "QNN_Reupload", "Backend": config.backend_name, "Shots": config.shots, "EvalSamples": n_test})
    rows.append(row)

    # VQC alternate
    vqc_n_train = min(config.vqc_train_size, len(X_train))
    vqc_n_test = min(config.vqc_test_size, len(X_test))
    if reporter is not None:
        reporter.step("VQC Hardware: training started")
    vqc_out = _train_torch_binary_qnn_hardware(
        qml, torch, nn, optim, DataLoader, TensorDataset,
        X_train[:vqc_n_train], y_train[:vqc_n_train], X_test[:vqc_n_test], config,
        template="vqc_alt",
        layers=config.vqc_layers,
        epochs=config.vqc_epochs,
        reporter=reporter
    )
    outputs["VQC_AltAnsatz"] = vqc_out
    row = evaluate_predictions(y_test[:vqc_n_test], vqc_out.hard, vqc_out.prob, config.target_fprs)
    row.update({"Model": "VQC_AltAnsatz", "Backend": config.backend_name, "Shots": config.shots, "EvalSamples": vqc_n_test})
    rows.append(row)

    return rows, outputs


def run_quantum_knn_hardware(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    config: HardwareConfig,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[List[dict], Dict[str, ModelOutput]]:
    """
    Run QKNN on hardware backend
    """
    if not HARDWARE_AVAILABLE:
        return [{"Model": "QKNN", "Status": f"SKIPPED: {IMPORT_ERROR}"}], {}

    if not config.run_qknn:
        if reporter is not None:
            reporter.step("QKNN skipped per configuration")
        return [], {}

    rows: List[dict] = []
    outputs: Dict[str, ModelOutput] = {}

    n_train = min(config.qknn_train_size, len(X_train))
    n_test = min(config.qknn_test_size, len(X_test))
    X_tr = X_train[:n_train]
    y_tr = y_train[:n_train]
    X_te = X_test[:n_test]
    y_te = y_test[:n_test]

    dev, _ = _get_hardware_device(config.num_features, config)

    def one_query(xq: np.ndarray) -> Tuple[int, float, float]:
        sims = np.array([
            _quantum_kernel_predict_hardware(qml, dev, xq, xt, config.num_features)
            for xt in X_tr
        ])
        top_idx = np.argsort(sims)[-config.qknn_k:]
        top_y = y_tr[top_idx]
        top_s = sims[top_idx]

        weighted_prob = float(np.sum(top_y * top_s) / max(np.sum(top_s), 1e-9))
        hard = int(weighted_prob >= 0.5)
        margin = weighted_prob * 2.0 - 1.0
        return hard, margin, weighted_prob

    if reporter is not None:
        reporter.step(f"QKNN Hardware: scoring on {config.backend_name} with {config.shots} shots")
    query_iter = X_te
    if reporter is not None:
        query_iter = reporter.loop(X_te, desc="QKNN HW queries", total=n_test)

    preds = [one_query(xq) for xq in query_iter]
    hard = np.array([p[0] for p in preds])
    margin = np.array([p[1] for p in preds])
    prob = np.array([p[2] for p in preds])

    outputs["QKNN"] = ModelOutput(hard=hard, margin=margin, prob=prob)
    row = evaluate_predictions(y_te, hard, prob, config.target_fprs)
    row.update({"Model": "QKNN", "Backend": config.backend_name, "Shots": config.shots, "EvalSamples": n_test})
    rows.append(row)

    return rows, outputs


def run_dataset_experiment_hardware(
    dataset_name: str,
    config: HardwareConfig,
    train_test_csv_path: str = "/home/ritvik/Python/Datasets/train_test_network.csv",
    cicids_folder_path: str = "/home/ritvik/Python/Datasets/CICIDS2017",
) -> pd.DataFrame:
    """
    Run complete quantum IDS experiment on hardware backend

    This is the main entry point for hardware experiments
    """
    warnings.filterwarnings("ignore")
    reporter = ProgressReporter(enabled=config.show_progress)
    ensure_dirs(config)
    set_publication_style(config.paper_fontsize)

    reporter.banner(f"Starting Hardware Experiment: {dataset_name}")
    reporter.step(f"Backend: {config.backend_name}, Shots: {config.shots}, Features: {config.num_features}")
    reporter.step(f"QNN models: {'enabled' if config.run_qnn_models else 'disabled'}, QKNN: {'enabled' if config.run_qknn else 'disabled'}")

    # Load data
    if dataset_name.startswith("train_test_network"):
        reporter.step("Loading train_test_network dataset")
        raw_df = load_train_test_network(train_test_csv_path)
        label_col = "label"
    elif dataset_name.startswith("cicids2017"):
        reporter.step("Loading CICIDS2017 dataset")
        raw_df = load_cicids(
            cicids_folder_path,
            sample_n=config.cicids_sample_n,
            seed=config.seed,
            max_files=config.cicids_max_files
        )
        label_col = "Label"
    else:
        raise ValueError("dataset_name must start with 'train_test_network' or 'cicids2017'")

    data = prepare_data(raw_df, config, label_col=label_col)
    reporter.step(f"Data prepared: train={len(data.X_train)}, test={len(data.X_test)}, dim={data.X_train.shape[1]}")

    all_rows: List[dict] = []
    all_outputs: Dict[str, ModelOutput] = {}

    # Classical baselines (same as simulator)
    reporter.banner("Classical baselines (hardware comparison)")
    classical_rows, classical_outputs = run_classical_baselines(
        data.X_train, data.y_train, data.X_test, data.y_test,
        config, reporter=reporter
    )
    all_rows.extend(classical_rows)
    all_outputs.update(classical_outputs)

    # Hardware quantum models
    reporter.banner("Quantum kernel model (QSVM) - HARDWARE")
    qsvm_rows, qsvm_outputs = run_qsvm_hardware(
        data.X_train, data.y_train, data.X_test, data.y_test,
        config, reporter=reporter
    )
    all_rows.extend(qsvm_rows)
    all_outputs.update(qsvm_outputs)

    if config.run_qnn_models:
        reporter.banner("Quantum neural models - HARDWARE")
        qnn_rows, qnn_outputs = run_quantum_nn_family_hardware(
            data.X_train, data.y_train, data.X_test, data.y_test,
            config, reporter=reporter
        )
        all_rows.extend(qnn_rows)
        all_outputs.update(qnn_outputs)

    if config.run_qknn:
        reporter.banner("Quantum-kNN - HARDWARE")
        qknn_rows, qknn_outputs = run_quantum_knn_hardware(
            data.X_train, data.y_train, data.X_test, data.y_test,
            config, reporter=reporter
        )
        all_rows.extend(qknn_rows)
        all_outputs.update(qknn_outputs)

    # Fusion schemes
    reporter.banner("Fusion sanity checks")
    fusion_rows = run_fusion_schemes(data.y_test, all_outputs, config, reporter=reporter)
    all_rows.extend(fusion_rows)

    # Plot export
    reporter.banner("Plot export")
    _plot_roc_pr(data.y_test, all_outputs, dataset_name, config, reporter=reporter)

    # Save results
    results_df = pd.DataFrame(all_rows)
    results_df.insert(0, "Dataset", dataset_name)

    per_dataset_path = os.path.join(config.output_root, f"{dataset_name}_metrics_summary.csv")
    results_df.to_csv(per_dataset_path, index=False)

    # Update consolidated table
    consolidated_path = os.path.join(config.output_root, "all_datasets_metrics_summary.csv")
    if os.path.exists(consolidated_path):
        prev = pd.read_csv(consolidated_path)
        merged = pd.concat([prev[prev["Dataset"] != dataset_name], results_df], ignore_index=True)
    else:
        merged = results_df.copy()
    merged.to_csv(consolidated_path, index=False)

    reporter.banner(f"Completed: {dataset_name}")
    reporter.step(f"Saved per-dataset summary: {per_dataset_path}")
    reporter.step(f"Updated combined summary: {consolidated_path}")

    return results_df
