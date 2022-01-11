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

f, ax = plt.subplots(1, len(runs), sharex=True, sharey=True)

for k, run in enumerate(runs):
    # Load data
    data = np.load(run['fn'])
    labels = data['labels']
    abnormality = np.exp(data['scores'])
    normality = np.exp(data['scores_normal'])

    # Plot scores for actual normals
    m_abnormality = np.ma.masked_array(abnormality, mask=labels)
    # m_normality = np.ma.masked_array(normality, mask=labels)
    ax[k].hist(m_abnormality, label="normals", alpha=0.5, bins=10)

    # Plot scores for actual abnormals
    m_abnormality = np.ma.masked_array(abnormality, mask=1-labels)
    # m_normality = np.ma.masked_array(normality, mask=1-labels)
    ax[k].hist(m_abnormality, label="abnormals", alpha=0.5, bins=10)

    ax[k].set_xlabel('p(abnormal)')
    ax[k].set_ylabel('frequency')
    ax[k].set_title(run['name'])

plt.legend()
plt.show()


