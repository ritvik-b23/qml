#!/usr/bin/env python3
"""
Generate confusion matrix PDF (cm3.pdf) with updated values
Matches the format of cm2.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Set paper-quality parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Confusion matrix data from CSV (MetaRF_Probability)
# D1 (CICIDS2017): TN=318, FP=5, FN=5, TP=22
# D2 (train_test_network): TN=85, FP=5, FN=0, TP=260

cm_d1 = np.array([[318, 5],
                  [5, 22]])

cm_d2 = np.array([[85, 5],
                  [0, 260]])

# Create figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Define labels
labels = ['Benign', 'Attack']

def plot_confusion_matrix(ax, cm, title):
    """Plot a single confusion matrix with annotations"""
    # Use coolwarm colormap for professional look
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')

# Plot both confusion matrices
plot_confusion_matrix(axes[0], cm_d1, 'D1')
plot_confusion_matrix(axes[1], cm_d2, 'D2')

# Adjust layout
plt.tight_layout()

# Save as PDF
output_file = 'cm3.pdf'
with PdfPages(output_file) as pdf:
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1)
    
print(f"✓ Generated {output_file}")
print(f"  D1 (CICIDS2017): TN={cm_d1[0,0]}, FP={cm_d1[0,1]}, FN={cm_d1[1,0]}, TP={cm_d1[1,1]}")
print(f"  D2 (train_test_network): TN={cm_d2[0,0]}, FP={cm_d2[0,1]}, FN={cm_d2[1,0]}, TP={cm_d2[1,1]}")

plt.close()
