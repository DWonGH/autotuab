# There should always be a 'train' and 'eval' folder directly
# below the first two given folders, containing all normal and abnormal data files without duplications (normally from
# TUAB). Folders after the first are used for training only when training_labels_from_csv is True.
data_folders = [
    # 'H:\TUAB2/normal/edf/',
    # 'H:\TUAB2/abnormal/edf/',
    'H:\TUAB2_relabelled/normal/edf/',
    'H:\TUAB2_relabelled/abnormal/edf/',
    'H:\TUEG/']
n_recordings = None  # set to an integer, if you want to restrict the set size
sensor_types = ["EEG"]
n_chans = 21
max_recording_mins = 35  # exclude larger recordings from training set
sec_to_cut = 60  # cut away at start of each recording
duration_recording_mins = 20  # how many minutes to use per recording
test_recording_mins = 20
max_abs_val = 800  # for clipping
sampling_freq = 100
divisor = 10  # divide signal by this
test_on_eval = True  # teston evaluation set or on training set
# in case of test on eval, n_folds and i_testfold determine
# validation fold in training set for training until first stop
n_folds = 10
i_test_fold = 9
shuffle = True
model_name = 'deep'
n_start_chans = 25
n_chan_factor = 2  # relevant for deep model only
input_time_length = 6000
final_conv_length = 1
model_constraint = 'defaultnorm'
init_lr = 1e-3
batch_size = 64
max_epochs = 10 # until first stop, the continue train on train+valid
cuda = True # False
training_labels_from_csv = True # Activates AutoTUAB (or enables TUAB+) training data
append_csv_set_to_TUAB = False # Activates TUAB+


def copy(target_list):
    target_list