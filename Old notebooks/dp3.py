import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek

df = pd.read_csv('IoT_Network_Intrusion_Dataset.csv', delimiter=',', encoding='utf-8')
df.info()

# DATA PREPROCESSING


columns_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Cat', 'Sub_Cat']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# replacing invalid values  with nan
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# replacing nans with the means of each column
df.fillna(df.mean(numeric_only=True), inplace=True)


print(df['Label'].unique())

# removing whitespaces from labels
# df['Label'] = df['Label'].str.strip()

# label encoding ('Anomaly' -> 0, 'Normal' -> 1)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

print("Encoded labels mapping:")
for idx, label in enumerate(le.classes_):
    print(f"{label} -> {idx}")
    

X = df.drop('Label', axis=1)
y = df['Label']


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_small, y_train_small = X_train_scaled[:200, :4], y_train[:200]
X_test_small, y_test_small   = X_test_scaled[:100, :4], y_test[:100]

# applying SMOTE to balance the training dataset

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_small, 
                                                              y_train_small)
print(f"Original class distribution: {np.bincount(y_train_small)}")
print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")

dev = qml.device("lightning.qubit", wires=4)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(4))
    for wire in range(4):
        qml.Hadamard(wires=wire)
    qml.AngleEmbedding(x2, wires=range(4))
    return qml.probs(wires=range(4))

def compute_kernel_matrix(X1, X2):
    """Compute the kernel matrix between two datasets."""
    n_samples_1, n_samples_2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n_samples_1, n_samples_2))

    print(f"Computing kernel matrix of size {n_samples_1}x{n_samples_2}...")
    
    for i in range(n_samples_1):
        if i % 10 == 0:
            print(f"Computing row {i+1}/{n_samples_1}...")
        for j in range(n_samples_2):
            kernel_matrix[i, j] = kernel_circuit(X1[i], X2[j])[0]

    print("Kernel matrix computation completed.")
    return kernel_matrix


print("Starting training kernel matrix computation...")
training_kernel_matrix = compute_kernel_matrix(X_train_balanced, X_train_balanced)

print("Starting testing kernel matrix computation...")
testing_kernel_matrix = compute_kernel_matrix(X_test_small, X_train_balanced)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

print("Training classical SVM with quantum kernel...")

# class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# grid search to get best C value
param_grid = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
svm_model = SVC(kernel="precomputed", class_weight=class_weights_dict)

grid_search = GridSearchCV(estimator=svm_model,
                           param_grid=param_grid,
                           scoring='f1_weighted',  # use F1-score for imbalanced datasets
                           cv=5,
                           verbose=1)

grid_search.fit(training_kernel_matrix, y_train_balanced)

# training the model using best C value from grid search
best_C = grid_search.best_params_['C']
svm_model_best = SVC(kernel="precomputed", C=best_C, class_weight=class_weights_dict)
svm_model_best.fit(training_kernel_matrix, y_train_balanced)

# predictions
y_pred_qsvc = svm_model_best.predict(testing_kernel_matrix)

# evaluation metrics
accuracy_qsvc = accuracy_score(y_test_small, y_pred_qsvc)
print(f"QSVM Accuracy on Test Data: {accuracy_qsvc:.4f}")

print("Classification Report:")
print(classification_report(y_test_small, y_pred_qsvc))

cm = confusion_matrix(y_test_small, y_pred_qsvc)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Anomaly", "Normal"], yticklabels=["Anomaly", "Normal"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# decision scores for ROC curve
y_score_qsvc = svm_model_best.decision_function(testing_kernel_matrix)

# calculate ROC curve and AUC
fpr_qsvc, tpr_qsvc, _ = roc_curve(y_test_small, y_score_qsvc)
roc_auc_qsvc = auc(fpr_qsvc, tpr_qsvc)


plt.figure(figsize=(8, 6))
plt.plot(fpr_qsvc, tpr_qsvc, color='darkorange', lw=2,
         label=f'QSVM ROC curve (AUC = {roc_auc_qsvc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (QSVM)')
plt.legend(loc="lower right")
plt.show()