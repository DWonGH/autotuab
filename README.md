# AutoTUAB

This repository contains code and data used to train Schirrmeister et al's BD-Deep4 convolutional neural network against the variants of the Temple University Hospital Abnormal EEG Corpus (TUAB). In particular, it allows the AutoTUAB dataset to be generated from the larger Temple University Hospital EEG Corpus (TUEG) using labels based on automated classification of the text reports accompanying each EEG session in that database.


# Reproducing Schirrmeister et al

This repo originated with an attempt to reproduce the work of Schirrmeister et al, by forking github.com/robintibor/auto-eeg-diagnosis-example.

To make that code work, this version has a legacy version of braindecode included. Hence if you have installed braindecode in your environment, you need to uninstall it for this to work.

If that's all you need, checkout the tag 'minimal':
```console
git checkout minimal
```

Beyond that tag, things deviate from robintibor's original example.

# Original README for robintibor/auto-eeg-diagnosis-example

## Requirements
1. Depends on https://robintibor.github.io/braindecode/ 
2. This code was programmed in Python 3.6 (might work for other versions also).

## Run
1. Modify config.py, especially correct data folders for your path..
2. Run with `python ./auto_diagnosis.py`

## Paper / Citation
The corresponding paper *Deep learning with convolutional neural networks for decoding and visualization of EEG pathology* is at https://arxiv.org/abs/1708.08012 . Now published at http://ieeexplore.ieee.org/document/8257015/ , please cite that version:

```
@INPROCEEDINGS{schirrmreegdiag2017,
  author={R. Schirrmeister and L. Gemein and K. Eggensperger and F. Hutter and T. Ball},
  booktitle={2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  title={Deep learning with convolutional neural networks for decoding and visualization of EEG pathology},
  year={2017},
  volume={},
  number={},
  pages={1-7},
  ISSN={},
  month={Dec},}
```
