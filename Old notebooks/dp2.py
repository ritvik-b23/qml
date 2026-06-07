import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("Loading datasets...")
train_df = pd.read_csv('NSL_KDD_Train.csv', header=None)
test_df = pd.read_csv('NSL_KDD_Test.csv', header=None)

print("Datasets loaded successfully.")

# Define column names based on NSL-KDD structure
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    'label'
]

train_df.columns = column_names
test_df.columns = column_names

print("Encoding categorical features...")
combined_df = pd.concat([train_df, test_df], axis=0)
categorical_cols = ['protocol_type', 'service', 'flag', 'label']
for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

train_df = combined_df.iloc[:len(train_df), :]
test_df = combined_df.iloc[len(train_df):, :]

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Subset data for quantum feasibility
X_train_small, y_train_small = X_train_scaled[:500, :4], y_train[:500]
X_test_small, y_test_small   = X_test_scaled[:100, :4], y_test[:100]

# # using the full dataset
# X_train_small, y_train_small = X_train_scaled, y_train
# X_test_small, y_test_small   = X_test_scaled, y_test
# output of lines 56 & 57: 
#       "ArrayMemoryError: Unable to allocate 118. GiB for an array with shape (125973, 125973) and data type float64" 💀😭


print("Data preprocessing completed.")


import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

dev = qml.device("lightning.qubit", wires=4)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(4))
    for wire in range(4):
        qml.Hadamard(wires=wire)
    qml.AngleEmbedding(x2, wires=range(4))
    return qml.probs(wires=range(4))

def compute_kernel_matrix(X1, X2):
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
training_kernel_matrix = compute_kernel_matrix(X_train_small, X_train_small)

print("Starting testing kernel matrix computation...")
testing_kernel_matrix = compute_kernel_matrix(X_test_small, X_train_small)

print("Training classical SVM with quantum kernel...")

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
svm_model = SVC(kernel="precomputed")
# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=svm_model,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1)

# Fit the model using the training kernel matrix
grid_search.fit(training_kernel_matrix, y_train_small)

print("Best C:", grid_search.best_params_['C'])
print("Best Accuracy:", grid_search.best_score_)


from sklearn.metrics import accuracy_score

# Train the model on full training set using best C value from GridSearchCV
best_C = grid_search.best_params_['C']
svm_model_best = SVC(kernel="precomputed", C=best_C)
svm_model_best.fit(training_kernel_matrix, y_train_small)

print("Making predictions on test data...")
y_pred_qsvc = svm_model_best.predict(testing_kernel_matrix)

# Evaluate accuracy on test data
accuracy_qsvc = accuracy_score(y_test_small, y_pred_qsvc)
print(f"QSVM Accuracy on Test Data: {accuracy_qsvc:.4f}")


from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Compute Confusion Matrix
cm = confusion_matrix(y_test_small, y_pred_qsvc)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test_small, y_pred_qsvc))

# Precision, Recall, F1-Score
precision = precision_score(y_test_small, y_pred_qsvc, average='weighted')
recall = recall_score(y_test_small, y_pred_qsvc, average='weighted')
f1 = f1_score(y_test_small, y_pred_qsvc, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# best_model = grid_search.best_estimator_
# test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
# print("Test set accuracy:", test_accuracy)

# # Print the best parameter and corresponding accuracy
# print("Best C:", grid_search.best_params_['C'])
# print("Best Accuracy:", grid_search.best_score_)

#----------------------------------------------------------

# svm_model = SVC(kernel="precomputed", C=10)
# print("Performing cross-validation...")
# cv_scores = cross_val_score(svm_model, training_kernel_matrix, y_train_small, cv=5)

# # Print cross-validation scores and mean accuracy
# print("Cross-validation scores:", cv_scores)
# print("Mean CV Accuracy:", np.mean(cv_scores))

# # Train the model using the training kernel matrix
# svm_model.fit(training_kernel_matrix, y_train_small)

# # Predict using the testing kernel matrix
# y_pred_qsvc = svm_model.predict(testing_kernel_matrix)

# # Evaluate accuracy
# accuracy_qsvc = accuracy_score(y_test_small, y_pred_qsvc)
# print(f"QSVM Accuracy: {accuracy_qsvc:.4f}")







# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
# svm_model = SVC(kernel="precomputed")
# # Perform grid search with cross-validation
# grid_search = GridSearchCV(estimator=svm_model,
#                            param_grid=param_grid,
#                            scoring='accuracy',
#                            cv=5,
#                            verbose=1)

# # Fit the model using the training kernel matrix
# grid_search.fit(training_kernel_matrix, y_train_small)

# # Print the best parameter and corresponding accuracy
# print("Best C:", grid_search.best_params_['C'])
# print("Best Accuracy:", grid_search.best_score_)
