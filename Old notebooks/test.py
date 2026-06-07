# import pandas as pd

# # Define correct column names
# column_names = [
#     "duration", "protocol_type", "service", "flag", "src_bytes", 
#     "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
#     "logged_in","num_compromised","root_shell","su_attempted","num_root",
#     "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
#     "is_host_login","is_guest_login","count","srv_count","serror_rate",
#     "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
#     "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
#     "dst_host_same_srv_rate","dst_host_diff_srv_rate",
#     "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
#     "dst_host_serror_rate","dst_host_srv_serror_rate",
#     "dst_host_rerror_rate","dst_host_srv_rerror_rate",
#     # target variable
#     'label'
# ]

# # Load datasets with proper headers
# train_df = pd.read_csv('NSL_KDD_Train.csv', header=None, names=column_names)
# test_df = pd.read_csv('NSL_KDD_Test.csv', header=None, names=column_names)

# # Verify columns
# print(train_df.head())
# print(train_df.columns)


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# print("Loading new dataset...")
# df = pd.read_csv('Small_IoT_Network_Intrusion_Dataset.csv')

# print(df.info())

# print("Unique values in 'Label' column:")
# print(df['Label'].unique())
# # Strip leading/trailing whitespace and standardize values
# df['Label'] = df['Label'].str.strip()

# # Replace unexpected values (if necessary)
# df['Label'] = df['Label'].replace({'Anomaly': 'Anomaly', 'Normal': 'Normal'})
# from sklearn.preprocessing import LabelEncoder

# # Encode the 'Label' column
# le = LabelEncoder()
# df['Label'] = le.fit_transform(df['Label'])

# print("Encoded labels:")
# print(le.classes_)


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# print("Loading new dataset...")
# df = pd.read_csv('Small_IoT_Network_Intrusion_Dataset.csv')

# print(df.info())

# print("Unique values in 'Label' column:")
# print(df['Label'].unique())
# # Strip leading/trailing whitespace and standardize values
# df['Label'] = df['Label'].str.strip()

# # Replace unexpected values (if necessary)
# df['Label'] = df['Label'].replace({'Anomaly': 'Anomaly', 'Normal': 'Normal'})
# from sklearn.preprocessing import LabelEncoder

# # Encode the 'Label' column
# le = LabelEncoder()
# df['Label'] = le.fit_transform(df['Label'])

# print("Encoded labels:")
# print(le.classes_)







# print("Dataset loaded successfully.")

# # Drop irrelevant columns (e.g., Flow_ID, Src_IP, etc.)
# columns_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Cat', 'Sub_Cat']
# df = df.drop(columns=columns_to_drop)

# # Check for invalid values and clean them
# print("Checking for invalid values...")
# df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.fillna(df.mean(), inplace=True)

# # Encode categorical features
# categorical_cols = ['Label']
# label_encoders = {}
# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le



# # Separate features and labels
# X = df.drop('Label', axis=1)
# y = df['Label']

# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale numerical features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Subset data for quantum feasibility (reduce rows and columns)
# X_train_small, y_train_small = X_train_scaled[:500, :4], y_train[:500]
# X_test_small, y_test_small   = X_test_scaled[:100, :4], y_test[:100]

# print("Data preprocessing completed.")




# import pandas as pd

# # Load the dataset
# df = pd.read_csv('Small_IoT_Network_Intrusion_Dataset.csv')

# # Check unique values in the Label column
# print("Unique values in 'Label' column:")
# print(df['Label'].unique())

# # Count the number of samples for each label
# label_counts = df['Label'].value_counts()

# print("\nNumber of samples for each label:")
# print(label_counts)

# # Specifically count Anomaly and Normal samples
# anomaly_count = label_counts.get('Anomaly', 0)
# normal_count = label_counts.get('Normal', 0)

# print(f"\nNumber of Anomaly samples: {anomaly_count}")
# print(f"Number of Normal samples: {normal_count}")




import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset with explicit delimiter and encoding for safety
print("Loading new dataset...")
df = pd.read_csv('Small_IoT_Network_Intrusion_Dataset.csv', delimiter=',', encoding='utf-8')

print("Dataset loaded successfully.")

# Drop irrelevant columns for modeling
columns_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Cat', 'Sub_Cat']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Check for invalid values (inf, -inf) and replace them with NaN
print("Cleaning invalid values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace NaNs with mean of each column
df.fillna(df.mean(numeric_only=True), inplace=True)

# Check again if Label column has correct unique values
print("Unique values in Label column before encoding:")
print(df['Label'].unique())

# Remove leading/trailing whitespace from labels (if any)
df['Label'] = df['Label'].str.strip()

# Encode labels properly ('Anomaly' -> 0, 'Normal' -> 1)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

print("Encoded labels mapping:")
for idx, label in enumerate(le.classes_):
    print(f"{label} -> {idx}")

# Separate features and labels
X = df.drop('Label', axis=1)
y = df['Label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Subset data for quantum feasibility (reduce rows and columns)
X_train_small, y_train_small = X_train_scaled[:700, :4], y_train[:700]
X_test_small, y_test_small   = X_test_scaled[:140, :4], y_test[:140]

print("Data preprocessing completed successfully.")



import pennylane as qml
import numpy as np

# Define a quantum device (simulator)
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
training_kernel_matrix = compute_kernel_matrix(X_train_small, X_train_small)

print("Starting testing kernel matrix computation...")
testing_kernel_matrix = compute_kernel_matrix(X_test_small, X_train_small)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("Balanced Training Dataset Size:")
print("Class Distribution:", np.bincount(y_train_balanced))



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
grid_search.fit(training_kernel_matrix, y_train_balanced)

# Print the best parameter and corresponding accuracy
print("Best C:", grid_search.best_params_['C'])
print("Best Accuracy:", grid_search.best_score_)


from sklearn.metrics import accuracy_score

# Train the model on full training set using best C value from GridSearchCV
best_C = grid_search.best_params_['C']
svm_model_best = SVC(kernel="precomputed", C=best_C, class_weight={0: 1, 1: 10})
svm_model_best.fit(training_kernel_matrix, y_train_balanced)

print("Making predictions on test data...")
y_pred_qsvc = svm_model_best.predict(testing_kernel_matrix)

# Evaluate accuracy on test data
accuracy_qsvc = accuracy_score(y_train_balanced, y_pred_qsvc)
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


# Count the number of samples classified as Anomaly (encoded as 0) and Normal (encoded as 1)
anomaly_count = np.sum(y_pred_qsvc == 0)
normal_count = np.sum(y_pred_qsvc == 1)

print(f"Number of samples classified as Anomaly: {anomaly_count}")
print(f"Number of samples classified as Normal: {normal_count}")

