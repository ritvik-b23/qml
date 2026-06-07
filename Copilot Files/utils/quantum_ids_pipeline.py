import glob
import importlib
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm


@dataclass
class ExperimentConfig:
    seed: int = 42
    test_size: float = 0.2
    num_features: int = 13
    max_workers: int = max(1, (os.cpu_count() or 4) // 2)
    target_fprs: Tuple[float, float] = (0.001, 0.01)

    cicids_sample_n: int = 20000
    cicids_max_files: Optional[int] = None

    qsvm_train_size: int = 300
    qsvm_test_size: int = 100
    qsvm_c_grid: Tuple[float, ...] = (0.1, 1.0, 10.0, 20.0, 30.0)

    qnn_train_size: int = 3000
    qnn_test_size: int = 1000
    qnn_layers: int = 5
    qnn_epochs: int = 35
    qnn_batch_size: int = 32
    qnn_lr: float = 0.01

    vqc_train_size: int = 2000
    vqc_test_size: int = 1000
    vqc_layers: int = 4
    vqc_epochs: int = 30

    qknn_train_size: int = 400
    qknn_test_size: int = 200
    qknn_k: int = 7

    export_dpi: int = 600
    paper_fontsize: int = 12

    output_root: str = "/home/ritvik/Python/Copilot Files/results"
    plots_pdf_dir: str = "/home/ritvik/Python/Plots PDF"
    images_dir: str = "/home/ritvik/Python/Images"

    show_progress: bool = True


class ProgressReporter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._start = time.time()

    def _prefix(self) -> str:
        elapsed = time.time() - self._start
        return f"[{elapsed:7.1f}s]"

    def banner(self, title: str) -> None:
        if not self.enabled:
            return
        bar = "=" * 76
        print(f"\n{bar}\n{title}\n{bar}")

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"{self._prefix()} {message}")

    def loop(self, iterable, desc: str, total: Optional[int] = None):
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            leave=False,
            disable=not self.enabled,
        )


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    pca: PCA
    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)


@dataclass
class ModelOutput:
    hard: np.ndarray
    margin: np.ndarray
    prob: np.ndarray


def ensure_dirs(config: ExperimentConfig) -> None:
    os.makedirs(config.output_root, exist_ok=True)
    os.makedirs(config.plots_pdf_dir, exist_ok=True)
    os.makedirs(config.images_dir, exist_ok=True)


def set_publication_style(fontsize: int = 12) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "figure.titlesize": fontsize,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_plot_pair(fig: plt.Figure, stem: str, config: ExperimentConfig) -> None:
    pdf_path = os.path.join(config.plots_pdf_dir, f"{stem}.pdf")
    png_path = os.path.join(config.images_dir, f"{stem}.png")
    jpg_path = os.path.join(config.images_dir, f"{stem}.jpg")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        png_path,
        format="png",
        dpi=config.export_dpi,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        jpg_path,
        format="jpg",
        dpi=config.export_dpi,
        bbox_inches="tight",
        pad_inches=0.05,
    )


def _normalize_label_series(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        y_norm = y.astype(str).str.strip().str.lower()
        return (y_norm != "benign").astype(int)
    y_unique = sorted(pd.Series(y).dropna().unique().tolist())
    if set(y_unique).issubset({0, 1}):
        return y.astype(int)
    # Fallback: map the most frequent label to 0, all else to 1.
    majority = y.value_counts().idxmax()
    return (y != majority).astype(int)


def load_train_test_network(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def load_cicids(folder_path: str, sample_n: int, seed: int, max_files: Optional[int]) -> pd.DataFrame:
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if max_files is not None:
        all_files = all_files[:max_files]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    # Memory-efficient loading: process files one at a time with sampling
    df_list = []
    sample_per_file = max(sample_n // len(all_files), 1000) if sample_n else None

    for idx, f in enumerate(all_files):
        try:
            # Use chunked reading for very large files
            if sample_per_file:
                # Read in chunks and sample from each chunk
                chunks = []
                chunk_size = 50000  # Read 50k rows at a time
                for chunk in pd.read_csv(f, low_memory=False, chunksize=chunk_size):
                    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
                    chunks.append(chunk)
                    # Early stopping if we have enough data
                    if sum(len(c) for c in chunks) >= sample_per_file * 2:
                        break

                if chunks:
                    temp_df = pd.concat(chunks, ignore_index=True)
                    if len(temp_df) > sample_per_file:
                        temp_df = temp_df.sample(n=min(sample_per_file, len(temp_df)), random_state=seed + idx)
                    df_list.append(temp_df)
            else:
                # Load full file if no sampling requested
                temp_df = pd.read_csv(f, low_memory=False)
                temp_df = temp_df.replace([np.inf, -np.inf], np.nan).dropna()
                df_list.append(temp_df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue

    if not df_list:
        raise ValueError("No data could be loaded from any CSV file")

    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns and " Label" in df.columns:
        df = df.rename(columns={" Label": "Label"})

    # Final sampling if we have more data than requested
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed)
    return df


def prepare_data(df: pd.DataFrame, config: ExperimentConfig, label_col: str = "label") -> DataBundle:
    if label_col not in df.columns:
        candidates = [c for c in df.columns if c.lower() == "label"]
        if candidates:
            label_col = candidates[0]
        else:
            raise ValueError(f"Label column not found. Available: {df.columns[:10]}")

    y = _normalize_label_series(df[label_col])
    X_df = df.drop(columns=[label_col]).copy()

    label_encoders: Dict[str, LabelEncoder] = {}
    categorical_cols = X_df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        label_encoders[col] = le

    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X_df,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    pca = PCA(n_components=config.num_features, random_state=config.seed)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train_s.to_numpy(),
        y_test=y_test_s.to_numpy(),
        scaler=scaler,
        pca=pca,
        label_encoders=label_encoders,
        feature_names=X_df.columns.tolist(),
    )


def compute_tpr_at_low_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fprs: Tuple[float, ...]) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out: Dict[str, float] = {}
    for tfpr in target_fprs:
        if tfpr <= 0:
            continue
        pct = tfpr * 100
        pct_label = f"{pct:g}"
        label = f"TPR@{pct_label}%FPR"
        out[label] = float(np.interp(tfpr, fpr, tpr))
    return out


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    target_fprs: Tuple[float, ...],
) -> Dict[str, float]:
    result = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUPRC": float(average_precision_score(y_true, y_score)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
    }
    result.update(compute_tpr_at_low_fpr(y_true, y_score, target_fprs))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return result


def _fit_grid_model(
    estimator,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    max_workers: int,
    reporter: Optional[ProgressReporter] = None,
    label: str = "GridSearch",
):
    if reporter is not None:
        reporter.step(f"{label}: CV tuning started.")
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="average_precision",
        cv=3,
        n_jobs=max_workers,
        refit=True,
    )
    grid.fit(X, y)
    if reporter is not None:
        reporter.step(f"{label}: best params = {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


def run_classical_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[List[dict], Dict[str, ModelOutput]]:
    rows: List[dict] = []
    outputs: Dict[str, ModelOutput] = {}

    # Logistic Regression
    if reporter is not None:
        reporter.step("Classical baseline: Logistic Regression")
    lr_base = LogisticRegression(max_iter=3000, solver="liblinear", random_state=config.seed)
    lr, lr_params = _fit_grid_model(
        lr_base,
        {"C": [0.1, 1.0, 10.0]},
        X_train,
        y_train,
        config.max_workers,
        reporter=reporter,
        label="LogReg",
    )
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_margin = lr.decision_function(X_test)
    lr_hard = lr.predict(X_test)
    outputs["LogReg"] = ModelOutput(hard=lr_hard, margin=lr_margin, prob=lr_prob)
    row = evaluate_predictions(y_test, lr_hard, lr_prob, config.target_fprs)
    row.update({"Model": "LogReg", "BestParams": str(lr_params)})
    rows.append(row)

    # Random Forest
    if reporter is not None:
        reporter.step("Classical baseline: Random Forest")
    rf_base = RandomForestClassifier(random_state=config.seed, n_jobs=config.max_workers)
    rf, rf_params = _fit_grid_model(
        rf_base,
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [1, 2],
        },
        X_train,
        y_train,
        config.max_workers,
        reporter=reporter,
        label="RandomForest",
    )
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_hard = rf.predict(X_test)
    rf_margin = rf_prob * 2.0 - 1.0
    outputs["RandomForest"] = ModelOutput(hard=rf_hard, margin=rf_margin, prob=rf_prob)
    row = evaluate_predictions(y_test, rf_hard, rf_prob, config.target_fprs)
    row.update({"Model": "RandomForest", "BestParams": str(rf_params)})
    rows.append(row)

    # XGBoost / LightGBM fallback
    boost_model = None
    boost_name = None
    boost_params = {}

    try:
        if reporter is not None:
            reporter.step("Classical baseline: XGBoost (fallback to LightGBM)")
        XGBClassifier = getattr(importlib.import_module("xgboost"), "XGBClassifier")

        xgb_base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config.seed,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=config.max_workers,
        )
        boost_model, boost_params = _fit_grid_model(
            xgb_base,
            {"max_depth": [4, 6], "learning_rate": [0.05, 0.1], "n_estimators": [200, 350]},
            X_train,
            y_train,
            config.max_workers,
            reporter=reporter,
            label="XGBoost",
        )
        boost_name = "XGBoost"
    except Exception:
        try:
            LGBMClassifier = getattr(importlib.import_module("lightgbm"), "LGBMClassifier")

            lgb_base = LGBMClassifier(
                objective="binary",
                random_state=config.seed,
                n_estimators=350,
                n_jobs=config.max_workers,
            )
            boost_model, boost_params = _fit_grid_model(
                lgb_base,
                {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1], "n_estimators": [250, 400]},
                X_train,
                y_train,
                config.max_workers,
                reporter=reporter,
                label="LightGBM",
            )
            boost_name = "LightGBM"
        except Exception:
            boost_model = None

    if boost_model is not None and boost_name is not None:
        prob = boost_model.predict_proba(X_test)[:, 1]
        hard = (prob >= 0.5).astype(int)
        margin = prob * 2.0 - 1.0
        outputs[boost_name] = ModelOutput(hard=hard, margin=margin, prob=prob)
        row = evaluate_predictions(y_test, hard, prob, config.target_fprs)
        row.update({"Model": boost_name, "BestParams": str(boost_params)})
        rows.append(row)
    else:
        if reporter is not None:
            reporter.step("Classical baseline: Boosting model skipped (xgboost/lightgbm unavailable)")
        rows.append(
            {
                "Model": "XGBoost/LightGBM",
                "Status": "SKIPPED: neither xgboost nor lightgbm importable",
            }
        )

    return rows, outputs


def _safe_import_quantum_stack():
    try:
        import pennylane as qml
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        return qml, torch, nn, optim, DataLoader, TensorDataset
    except Exception as exc:
        return None, None, None, None, None, None


def _quantum_kernel_predict(qml, dev, x1: np.ndarray, x2: np.ndarray, num_features: int) -> float:
    @qml.qnode(dev)
    def kernel_circuit(a, b):
        qml.AngleEmbedding(a, wires=range(num_features))
        qml.adjoint(qml.AngleEmbedding)(b, wires=range(num_features))
        return qml.probs(wires=range(num_features))

    return float(kernel_circuit(x1, x2)[0])


def _quantum_kernel_matrix(
    qml,
    dev,
    X1: np.ndarray,
    X2: np.ndarray,
    num_features: int,
    max_workers: int,
    reporter: Optional[ProgressReporter] = None,
    desc: str = "Kernel rows",
) -> np.ndarray:
    n1, n2 = len(X1), len(X2)

    def one_row(i: int) -> np.ndarray:
        row = np.zeros(n2)
        for j in range(n2):
            row[j] = _quantum_kernel_predict(qml, dev, X1[i], X2[j], num_features)
        return row

    row_iter = range(n1)
    if reporter is not None:
        row_iter = reporter.loop(row_iter, desc=desc, total=n1)
    rows = Parallel(n_jobs=max_workers)(delayed(one_row)(i) for i in row_iter)
    return np.vstack(rows)


def run_qsvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[List[dict], Dict[str, ModelOutput]]:
    qml, *_ = _safe_import_quantum_stack()
    rows: List[dict] = []
    outputs: Dict[str, ModelOutput] = {}

    if qml is None:
        rows.append({"Model": "QSVM", "Status": "SKIPPED: pennylane/torch stack unavailable"})
        return rows, outputs

    # Use same total samples as other models (3000 train, 1000 test)
    # Split into 10 batches of 300/100 to keep computation manageable
    n_batches = 10
    batch_train_size = config.qsvm_train_size  # 300 per batch
    batch_test_size = config.qsvm_test_size    # 100 per batch
    total_train = n_batches * batch_train_size  # 3000
    total_test = n_batches * batch_test_size    # 1000

    n_train = min(total_train, len(X_train))
    n_test = min(total_test, len(X_test))

    X_all_train = X_train[:n_train]
    y_all_train = y_train[:n_train]
    X_all_test = X_test[:n_test]
    y_all_test = y_test[:n_test]

    dev = qml.device("default.qubit", wires=config.num_features)

    # Collect predictions from all batches
    all_hard = []
    all_margin = []
    all_prob = []
    best_c_values = []

    if reporter is not None:
        reporter.step(f"QSVM: processing {n_batches} batches ({batch_train_size} train, {batch_test_size} test per batch)")

    batch_iter = range(n_batches)
    if reporter is not None:
        batch_iter = reporter.loop(batch_iter, desc="QSVM batches", total=n_batches)

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
        K_train = _quantum_kernel_matrix(
            qml,
            dev,
            X_tr,
            X_tr,
            config.num_features,
            max_workers=1,
            reporter=None,  # Suppress progress for each batch
            desc=f"QSVM batch {batch_idx+1} train kernel",
        )

        K_test = _quantum_kernel_matrix(
            qml,
            dev,
            X_te,
            X_tr,
            config.num_features,
            max_workers=1,
            reporter=None,  # Suppress progress for each batch
            desc=f"QSVM batch {batch_idx+1} test kernel",
        )

        # Train SVC with grid search for this batch
        svc = SVC(kernel="precomputed", probability=True)
        grid = GridSearchCV(
            svc,
            {"C": list(config.qsvm_c_grid)},
            cv=3,
            scoring="average_precision",
            n_jobs=1,
        )
        grid.fit(K_train, y_tr)
        best = grid.best_estimator_
        best_c_values.append(grid.best_params_['C'])

        # Get predictions for this batch
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

    # Report the most common C value across batches
    from collections import Counter
    most_common_c = Counter(best_c_values).most_common(1)[0][0]
    row.update({
        "Model": "QSVM",
        "BestParams": f"C={most_common_c} (mode across {n_batches} batches)",
        "EvalSamples": n_test
    })
    rows.append(row)

    if reporter is not None:
        reporter.step(f"QSVM: completed {n_batches} batches, evaluated on {n_test} samples")

    return rows, outputs


def _train_torch_binary_qnn(
    qml,
    torch,
    nn,
    optim,
    DataLoader,
    TensorDataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: ExperimentConfig,
    template: str,
    layers: int,
    epochs: int,
    reporter: Optional[ProgressReporter] = None,
) -> ModelOutput:
    n_qubits = config.num_features
    dev = qml.device("default.qubit", wires=n_qubits)

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
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.qnn_lr)

    model.train()
    epoch_iter = range(epochs)
    if reporter is not None:
        epoch_iter = reporter.loop(epoch_iter, desc=f"{template} epochs", total=epochs)
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


def run_quantum_nn_family(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[List[dict], Dict[str, ModelOutput]]:
    qml, torch, nn, optim, DataLoader, TensorDataset = _safe_import_quantum_stack()
    rows: List[dict] = []
    outputs: Dict[str, ModelOutput] = {}

    if qml is None:
        rows.append({"Model": "QNN/VQC", "Status": "SKIPPED: pennylane/torch stack unavailable"})
        return rows, outputs

    n_train = min(config.qnn_train_size, len(X_train))
    n_test = min(config.qnn_test_size, len(X_test))
    X_tr = X_train[:n_train]
    y_tr = y_train[:n_train]
    X_te = X_test[:n_test]
    y_te = y_test[:n_test]

    # Base QNN
    if reporter is not None:
        reporter.step(f"QNN base: training on {n_train} samples for {config.qnn_epochs} epochs")
    qnn_out = _train_torch_binary_qnn(
        qml,
        torch,
        nn,
        optim,
        DataLoader,
        TensorDataset,
        X_tr,
        y_tr,
        X_te,
        config,
        template="strongly_entangling",
        layers=config.qnn_layers,
        epochs=config.qnn_epochs,
        reporter=reporter,
    )
    outputs["QNN_Base"] = qnn_out
    row = evaluate_predictions(y_te, qnn_out.hard, qnn_out.prob, config.target_fprs)
    row.update({"Model": "QNN_Base", "EvalSamples": n_test})
    rows.append(row)

    # Data re-uploading QNN
    if reporter is not None:
        reporter.step("QNN re-uploading: training started")
    reup_out = _train_torch_binary_qnn(
        qml,
        torch,
        nn,
        optim,
        DataLoader,
        TensorDataset,
        X_tr,
        y_tr,
        X_te,
        config,
        template="reupload",
        layers=config.qnn_layers,
        epochs=max(20, config.qnn_epochs - 5),
        reporter=reporter,
    )
    outputs["QNN_Reupload"] = reup_out
    row = evaluate_predictions(y_te, reup_out.hard, reup_out.prob, config.target_fprs)
    row.update({"Model": "QNN_Reupload", "EvalSamples": n_test})
    rows.append(row)

    # VQC alternate ansatz
    vqc_n_train = min(config.vqc_train_size, len(X_train))
    vqc_n_test = min(config.vqc_test_size, len(X_test))
    if reporter is not None:
        reporter.step("VQC alternate ansatz: training started")
    vqc_out = _train_torch_binary_qnn(
        qml,
        torch,
        nn,
        optim,
        DataLoader,
        TensorDataset,
        X_train[:vqc_n_train],
        y_train[:vqc_n_train],
        X_test[:vqc_n_test],
        config,
        template="vqc_alt",
        layers=config.vqc_layers,
        epochs=config.vqc_epochs,
        reporter=reporter,
    )
    outputs["VQC_AltAnsatz"] = vqc_out
    row = evaluate_predictions(y_test[:vqc_n_test], vqc_out.hard, vqc_out.prob, config.target_fprs)
    row.update({"Model": "VQC_AltAnsatz", "EvalSamples": vqc_n_test})
    rows.append(row)

    return rows, outputs


def run_quantum_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
    reporter: Optional[ProgressReporter] = None,
) -> Tuple[List[dict], Dict[str, ModelOutput]]:
    qml, *_ = _safe_import_quantum_stack()
    rows: List[dict] = []
    outputs: Dict[str, ModelOutput] = {}

    if qml is None:
        rows.append({"Model": "QKNN", "Status": "SKIPPED: pennylane unavailable"})
        return rows, outputs

    n_train = min(config.qknn_train_size, len(X_train))
    n_test = min(config.qknn_test_size, len(X_test))
    X_tr = X_train[:n_train]
    y_tr = y_train[:n_train]
    X_te = X_test[:n_test]
    y_te = y_test[:n_test]

    dev = qml.device("default.qubit", wires=config.num_features)

    def one_query(xq: np.ndarray) -> Tuple[int, float, float]:
        sims = np.array([
            _quantum_kernel_predict(qml, dev, xq, xt, config.num_features) for xt in X_tr
        ])
        top_idx = np.argsort(sims)[-config.qknn_k :]
        top_y = y_tr[top_idx]
        top_s = sims[top_idx]

        weighted_prob = float(np.sum(top_y * top_s) / max(np.sum(top_s), 1e-9))
        hard = int(weighted_prob >= 0.5)
        margin = weighted_prob * 2.0 - 1.0
        return hard, margin, weighted_prob

    if reporter is not None:
        reporter.step(f"QKNN: scoring {n_test} queries against {n_train} references")
    query_iter = X_te
    if reporter is not None:
        query_iter = reporter.loop(X_te, desc="QKNN queries", total=n_test)
    preds = [one_query(xq) for xq in query_iter]
    hard = np.array([p[0] for p in preds])
    margin = np.array([p[1] for p in preds])
    prob = np.array([p[2] for p in preds])

    outputs["QKNN"] = ModelOutput(hard=hard, margin=margin, prob=prob)
    row = evaluate_predictions(y_te, hard, prob, config.target_fprs)
    row.update({"Model": "QKNN", "EvalSamples": n_test})
    rows.append(row)
    return rows, outputs


def _slice_to_common_length(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    n = min(len(a), len(b), len(c), len(d))
    return a[:n], b[:n], c[:n], d[:n]


def run_fusion_schemes(
    y_test: np.ndarray,
    outputs: Dict[str, ModelOutput],
    config: ExperimentConfig,
    model_a: str = "QSVM",
    model_b: str = "QNN_Base",
    reporter: Optional[ProgressReporter] = None,
) -> List[dict]:
    rows: List[dict] = []

    if model_a not in outputs or model_b not in outputs:
        rows.append(
            {
                "Model": "MetaRF_Fusion",
                "Status": f"SKIPPED: required outputs missing ({model_a}, {model_b})",
            }
        )
        return rows

    out_a = outputs[model_a]
    out_b = outputs[model_b]

    hard_a, hard_b, y_eval, _ = _slice_to_common_length(out_a.hard, out_b.hard, y_test, y_test)
    margin_a, margin_b, y_eval2, _ = _slice_to_common_length(out_a.margin, out_b.margin, y_test, y_test)
    prob_a, prob_b, y_eval3, _ = _slice_to_common_length(out_a.prob, out_b.prob, y_test, y_test)

    # Build distinct feature matrices by design.
    X_hard = np.column_stack([hard_a, hard_b])
    X_margin = np.column_stack([margin_a, margin_b])
    X_prob = np.column_stack([prob_a, prob_b])

    # Numerical-distinctness guardrails.
    if np.allclose(X_hard, X_margin):
        raise RuntimeError("Fusion bug: hard and margin features are numerically identical.")
    if np.allclose(X_hard, X_prob):
        raise RuntimeError("Fusion bug: hard and prob features are numerically identical.")
    if np.allclose(X_margin, X_prob):
        raise RuntimeError("Fusion bug: margin and prob features are numerically identical.")

    splits = {
        "Hard": (X_hard, y_eval),
        "Margin": (X_margin, y_eval2),
        "Probability": (X_prob, y_eval3),
    }

    scheme_metrics = {}
    scheme_iter = splits.items()
    if reporter is not None:
        scheme_iter = reporter.loop(list(splits.items()), desc="Fusion schemes", total=len(splits))
    for scheme_name, (X_scheme, y_scheme) in scheme_iter:
        if reporter is not None:
            reporter.step(f"Fusion {scheme_name}: training meta-RF")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scheme,
            y_scheme,
            test_size=0.35,
            random_state=config.seed,
            stratify=y_scheme,
        )

        meta = RandomForestClassifier(
            n_estimators=300,
            random_state=config.seed,
            n_jobs=config.max_workers,
        )
        meta.fit(X_tr, y_tr)

        prob = meta.predict_proba(X_te)[:, 1]
        hard = (prob >= 0.5).astype(int)
        metrics = evaluate_predictions(y_te, hard, prob, config.target_fprs)
        metrics.update({"Model": f"MetaRF_{scheme_name}", "FusionScheme": scheme_name})
        rows.append(metrics)
        scheme_metrics[scheme_name] = metrics

    # If all fusion schemes converge to same F1/AUPRC, annotate why.
    f1s = [round(scheme_metrics[s]["F1"], 6) for s in ["Hard", "Margin", "Probability"]]
    auprcs = [round(scheme_metrics[s]["AUPRC"], 6) for s in ["Hard", "Margin", "Probability"]]
    if len(set(f1s)) == 1 and len(set(auprcs)) == 1:
        note = (
            "Identical fusion metrics observed. Likely explanation: RF threshold splits are rank-driven; "
            "if margin/prob are monotonic transforms of near-binary signals, extra magnitude information may be ignored."
        )
        for row in rows:
            if row.get("Model", "").startswith("MetaRF_"):
                row["FusionAnalysis"] = note

    return rows


def _plot_roc_pr(
    y_test: np.ndarray,
    outputs: Dict[str, ModelOutput],
    dataset_name: str,
    config: ExperimentConfig,
    reporter: Optional[ProgressReporter] = None,
) -> None:
    if reporter is not None:
        reporter.step("Plotting ROC/PR comparisons and exporting PDF + PNG + JPG")
    # ROC plot
    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))
    for model_name, out in outputs.items():
        n = min(len(y_test), len(out.prob))
        if n < 10:
            continue
        y = y_test[:n]
        p = out.prob[:n]
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = roc_auc_score(y, p)
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Comparison - {dataset_name}")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    save_plot_pair(fig_roc, f"{dataset_name}_roc_comparison", config)

    # PR plot
    fig_pr, ax_pr = plt.subplots(figsize=(8, 5))
    for model_name, out in outputs.items():
        n = min(len(y_test), len(out.prob))
        if n < 10:
            continue
        y = y_test[:n]
        p = out.prob[:n]
        prec, rec, _ = precision_recall_curve(y, p)
        auprc = average_precision_score(y, p)
        ax_pr.plot(rec, prec, linewidth=2, label=f"{model_name} (AUPRC={auprc:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"PR Comparison - {dataset_name}")
    ax_pr.legend(loc="lower left")
    fig_pr.tight_layout()
    save_plot_pair(fig_pr, f"{dataset_name}_pr_comparison", config)

    plt.close(fig_roc)
    plt.close(fig_pr)


def run_dataset_experiment(
    dataset_name: str,
    config: ExperimentConfig,
    train_test_csv_path: str = "/home/ritvik/Python/Datasets/train_test_network.csv",
    cicids_folder_path: str = "/home/ritvik/Python/Datasets/CICIDS2017",
) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    reporter = ProgressReporter(enabled=config.show_progress)
    ensure_dirs(config)
    set_publication_style(config.paper_fontsize)
    reporter.banner(f"Starting experiment: {dataset_name}")
    reporter.step(f"max_workers={config.max_workers}, seed={config.seed}, n_features={config.num_features}")

    if dataset_name == "train_test_network":
        reporter.step("Loading train_test_network dataset")
        raw_df = load_train_test_network(train_test_csv_path)
        label_col = "label"
    elif dataset_name == "cicids2017":
        reporter.step("Loading CICIDS2017 dataset")
        raw_df = load_cicids(
            cicids_folder_path,
            sample_n=config.cicids_sample_n,
            seed=config.seed,
            max_files=config.cicids_max_files,
        )
        label_col = "Label"
    else:
        raise ValueError("dataset_name must be one of: train_test_network, cicids2017")

    data = prepare_data(raw_df, config, label_col=label_col)
    reporter.step(
        f"Data prepared: train={len(data.X_train)}, test={len(data.X_test)}, dim={data.X_train.shape[1]}"
    )

    all_rows: List[dict] = []
    all_outputs: Dict[str, ModelOutput] = {}

    reporter.banner("Classical baselines")
    classical_rows, classical_outputs = run_classical_baselines(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        config,
        reporter=reporter,
    )
    all_rows.extend(classical_rows)
    all_outputs.update(classical_outputs)

    reporter.banner("Quantum kernel model (QSVM)")
    qsvm_rows, qsvm_outputs = run_qsvm(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        config,
        reporter=reporter,
    )
    all_rows.extend(qsvm_rows)
    all_outputs.update(qsvm_outputs)

    reporter.banner("Quantum neural models")
    qnn_rows, qnn_outputs = run_quantum_nn_family(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        config,
        reporter=reporter,
    )
    all_rows.extend(qnn_rows)
    all_outputs.update(qnn_outputs)

    reporter.banner("Quantum-kNN")
    qknn_rows, qknn_outputs = run_quantum_knn(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        config,
        reporter=reporter,
    )
    all_rows.extend(qknn_rows)
    all_outputs.update(qknn_outputs)

    reporter.banner("Fusion sanity checks")
    fusion_rows = run_fusion_schemes(data.y_test, all_outputs, config, reporter=reporter)
    all_rows.extend(fusion_rows)

    reporter.banner("Plot export")
    _plot_roc_pr(data.y_test, all_outputs, dataset_name, config, reporter=reporter)

    results_df = pd.DataFrame(all_rows)
    results_df.insert(0, "Dataset", dataset_name)

    per_dataset_path = os.path.join(config.output_root, f"{dataset_name}_metrics_summary.csv")
    results_df.to_csv(per_dataset_path, index=False)

    # Update consolidated table.
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
