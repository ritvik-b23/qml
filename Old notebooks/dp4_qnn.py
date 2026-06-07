from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pennylane as qml
import torch
from torch import nn
from torch.optim import Adam
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

print("Loading new dataset...")
df = pd.read_csv('Small_IoT_Network_Intrusion_Dataset.csv', delimiter=',', encoding='utf-8')

print("Dataset loaded successfully.")

columns_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Cat', 'Sub_Cat']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print("Cleaning invalid values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

print("Unique values in Label column before encoding:")
print(df['Label'].unique())

df['Label'] = df['Label'].str.strip()

# ('Anomaly' -> 0, 'Normal' -> 1)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

print("Encoded labels mapping:")
for idx, label in enumerate(le.classes_):
    print(f"{label} -> {idx}")

X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_small, y_train_small = X_train_scaled[:200, :4], y_train[:200]
X_test_small, y_test_small   = X_test_scaled[:100, :4], y_test[:100]

# Reduce features to 4 for quantum feasibility
n_features = 4  # Number of qubits required
X_train_reduced = X_train_small[:, :n_features]
X_test_reduced = X_test_small[:, :n_features]


# SMOTE
smote = SMOTE(random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train_reduced, y_train)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_reduced, y_train_small)


print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")


print("Data preprocessing completed successfully.")


# Define Quantum Device
n_qubits = n_features  # Match number of qubits to features
dev = qml.device("lightning.qubit", wires=n_qubits)

# Define Quantum Circuit (Variational Quantum Circuit)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Encode classical data into quantum states
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    # Add trainable variational layers (Strongly Entangling Layers)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measure expectation values of Pauli-Z operators for each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define PyTorch Layer for Quantum Circuit
class QNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(3, n_qubits, 3))  # Adjust as needed

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        quantum_output = torch.stack([torch.tensor(quantum_circuit(inputs[i], self.weights)) for i in range(batch_size)])
        return quantum_output.view(batch_size, -1)  # Flatten to match Linear layer input


# Define Hybrid Quantum-Classical Neural Network Model
class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn_layer = QNNLayer()
        self.classifier = nn.Linear(n_qubits, 2)  # Ensure n_qubits matches quantum circuit output size

    def forward(self, x):
        qnn_output = self.qnn_layer(x)
        return self.classifier(qnn_output)



# X_train_torch = torch.tensor(X_train_reduced, dtype=torch.float32)
# y_train_torch = torch.tensor(y_train_small.values, dtype=torch.long)
X_train_torch = torch.tensor(X_train_balanced, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_balanced.values, dtype=torch.long)
X_test_torch = torch.tensor(X_test_reduced, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_small.values, dtype=torch.long)


# Compute class weights for cost-sensitive learning
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)



# Initialize model, loss function, and optimizer
model = HybridQNN()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training Loop
n_epochs = 30
batch_size = 32

print("Starting training...")
for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(X_train_torch.size(0))
    for i in range(0, X_train_torch.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_torch[indices], y_train_torch[indices]
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

print("Training completed.")


# Evaluate on Test Data
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_torch)
    y_pred_classes = torch.argmax(y_pred_probs, dim=1).numpy()

# Compute Metrics
accuracy = accuracy_score(y_test_torch.numpy(), y_pred_classes)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test_torch.numpy(), y_pred_classes))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test_torch.numpy(), y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

