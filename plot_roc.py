# Plot ROCs for comparison.
# It is assumed that the scores and labels (predictions and targets) have been saved in npz files already.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

runs = [{'name': 'AutoTUAB', 'fn':'Auto Diagnosis ROC File 2021-11-29 20_11_03.npz'},
        {'name': 'TUAB', 'fn':'Auto Diagnosis ROC File 2021-11-29 20_37_26.npz'}]
# runs = [{'name': 'AutoTUAB', 'fn':'Auto Diagnosis ROC File 2021-11-29 20_10_44.npz'},
#         {'name': 'TUAB', 'fn':'Auto Diagnosis ROC File 2021-11-29 20_37_07.npz'}]

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
    plt.plot(fpr, tpr, label=f"{run['name']}, auc={roc_auc}")

plt.legend()
plt.xlabel('1-specificity (FPR)')
plt.ylabel('sensitivity (FPR)')
plt.show()