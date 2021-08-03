# Copy this file as config_overrider.py and use it to set up parameters for batch processing.
# Each row represents one run. Each item in a row is a parameter as listed/set in config.py.

# Use the following to run a basic test of Schirrmeister et al's network configurations as explored in
# [1] R. T. Schirrmeister, L. Gemein, K. Eggensperger, F. Hutter, and T. Ball, ‘Deep learning with
#       convolutional neural networks for decoding and visualization of EEG pathology’, arXiv e-prints,
#       p. arXiv:1708.08012, Aug. 2017.
# new_config_values = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "shallow", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "deep", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "deep_smac", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "shallow_smac", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "linear", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Use the following values to run 5 repeats of training against AutoTUAB labellings.
new_config_values = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, "deep", 0, 0, 0, 0, 0, 0, 0, 0, 0, True, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, "deep", 0, 0, 0, 0, 0, 0, 0, 0, 0, True, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, "deep", 0, 0, 0, 0, 0, 0, 0, 0, 0, True, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, "deep", 0, 0, 0, 0, 0, 0, 0, 0, 0, True, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, "deep", 0, 0, 0, 0, 0, 0, 0, 0, 0, True, 0]]