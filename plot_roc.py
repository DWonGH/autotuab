# Plot ROCs for comparison.
# It is assumed that the scores and labels (predictions and targets) have been saved in npz files already.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

runs = [{'name': 'AutoTUAB', 'fn':'Auto Diagnosis ROC File 2021-11-30 16_30_21.npz'},
        {'name': 'TUAB', 'fn':'Auto Diagnosis ROC File 2021-12-02 12_02_46.npz'}] # Train
# runs = [{'name': 'AutoTUAB', 'fn':'Auto Diagnosis ROC File 2021-11-30 16_30_41.npz'},
#         {'name': 'TUAB', 'fn':'Auto Diagnosis ROC File 2021-12-02 12_03_05.npz'}] # Test
# runs = [{'name': 'AutoTUAB', 'fn':'Auto Diagnosis ROC File 2021-11-30 17_04_45.npz'},
#         {'name': 'TUAB', 'fn':'Auto Diagnosis ROC File 2021-12-02 12_03_05.npz'}] # Test

for run in runs:
    # Load data
    data = np.load(run['fn'])
    y = data['labels']
    pred = data['scores']

    # Calculate and plot ROC
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    # display.plot()
    # plt.plot(fpr, tpr, label=f"{run['name']}, auc={roc_auc}")
    plt.plot(fpr, tpr, label=f"{run['name']}")

plt.legend()
plt.xlabel('1-specificity (FPR)')
plt.ylabel('sensitivity (FPR)')
plt.show()