import logging
import re
import numpy as np
import glob
import os.path
from os import sep
import csv
import random

import mne

log = logging.getLogger(__name__)


def session_key(file_name):
    """ sort the file name by session """
    return re.findall(r'(s\d{2})', file_name)


def natural_key(file_name):
    """ provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None
           for token in re.split(r'(\d+)', file_name)]
    return key

def time_key(file_name):
    """ provides a time-based sorting key """
    splits = file_name.split(sep)
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])

    return date_id + session_id + recording_id


def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)

    if key == 'time':
        return sorted(file_paths, key=time_key)

    elif key == 'natural':
        return sorted(file_paths, key=natural_key)

    else:
        return file_paths

def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None
        # fix_header(file_path)
        # try:
        #     edf_file = mne.io.read_raw_edf(file_path, verbose='error')
        #     logging.warning("Fixed it!")
        # except ValueError:
        #     return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    # TODO: return rec object?
    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration


def get_recording_length(file_path):
    """ some recordings were that huge that simply opening them with mne caused the program to crash. therefore, open
    the edf as bytes and only read the header. parse the duration from there and check if the file can safely be opened
    :param file_path: path of the directory
    :return: the duration of the recording
    """
    f = open(file_path, 'rb')
    header = f.read(256)
    f.close()

    return int(header[236:244].decode('ascii'))


def load_data(fname, preproc_functions, sensor_types=['EEG']):
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)
    log.debug("Load data...")
    cnt.load_data()
    selected_ch_names = []
    if 'EEG' in sensor_types:
        wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                        'FP2', 'FZ', 'O1', 'O2',
                        'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

        for wanted_part in wanted_elecs:
            wanted_found_name = []
            for ch_name in cnt.ch_names:
                # if ' ' + wanted_part + '-' in ch_name:
                if wanted_part.lower() in ch_name.lower():
                    wanted_found_name.append(ch_name)
            # assert len(wanted_found_name) == 1
            if len(wanted_found_name) < 1:
                log.info("Desired electrodes not found. Skipping this file.")
                return None
            selected_ch_names.append(wanted_found_name[0])
    if 'EKG' in sensor_types:
        wanted_found_name = []
        for ch_name in cnt.ch_names:
            if 'EKG' in ch_name:
                wanted_found_name.append(ch_name)
        assert len(wanted_found_name) == 1
        selected_ch_names.append(wanted_found_name[0])

    cnt = cnt.pick_channels(selected_ch_names, ordered=True)

    assert np.array_equal(cnt.ch_names, selected_ch_names)
    n_sensors = 0
    if 'EEG' in sensor_types:
        n_sensors += 21
    if 'EKG' in sensor_types:
        n_sensors += 1

    assert len(cnt.ch_names)  == n_sensors, (
        "Expected {:d} channel names, got {:d} channel names".format(
            n_sensors, len(cnt.ch_names)))

    # change from volt to mikrovolt
    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    log.debug("Preprocessing...")
    for fn in preproc_functions:
        log.debug(fn)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)
    return data


def get_all_sorted_file_names_and_labels(train_or_eval, folders, training_labels_from_csv, append_csv_set_to_TUAB=False):
    all_file_names = []

    if append_csv_set_to_TUAB and not training_labels_from_csv:
        error("If append_csv_set_to_TUAB is set to True in config.py, training_labels_from_csv must also be set to True.")

    if training_labels_from_csv:
        TUEG_folders = folders[2:]

    if train_or_eval == 'eval' or append_csv_set_to_TUAB or not training_labels_from_csv:
        folders = folders[:2]
        folders = [os.path.join(folder, train_or_eval) + '/' for folder in folders]
        for full_folder in folders:
            log.info("Reading {:s}...".format(full_folder))
            this_file_names = read_all_file_names(full_folder, '.edf', key='time')
            log.info(".. {:d} files.".format(len(this_file_names)))
            all_file_names.extend(this_file_names)

        all_file_names = sorted(all_file_names, key=time_key)
        labels = [int('/abnormal/' in f) for f in all_file_names]


    if train_or_eval == 'train' and training_labels_from_csv:
        # Generate AutoTUAB/TUAB+ dataset

        with open('training_labels.csv', newline='') as csvfile:
            # label_catalog_reader = csv.reader(csvfile, delimiter='\t')
            label_catalog_reader = csv.reader(csvfile)

            # Skip the header row (column names)
            next(label_catalog_reader, None)

            all_labelled_TUEG_file_names = []
            TUEG_labels = []
            for row in label_catalog_reader:
                # Skip blank lines
                if len(row)==0:
                    continue
                id, _ = os.path.splitext(os.path.basename(row[1]))
                p_ab = float(row[2])
                label_from_ML = row[3]
                label_from_rules = row[4]
                comprehensive_decision = row[5]
                if label_from_rules!=2:
                    label = label_from_rules
                elif label_from_rules==2 and (p_ab>=0.99 or p_ab<=0.01):
                #if (p_ab>=0.99 or p_ab<=0.01):
                    label = label_from_ML
                else:
                    continue
                full_folder = os.path.join(TUEG_folders[0], row[0])
                full_folder = full_folder.replace('/TUEG_txt','')
                this_file_names = read_all_file_names(full_folder, '.edf', key='time')
                [all_labelled_TUEG_file_names.append(ff) for ff in this_file_names if id in os.path.basename(ff)]
                [TUEG_labels.append(label) for ff in this_file_names if id in os.path.basename(ff)]


        if append_csv_set_to_TUAB:
            # Join TUAB and TUAB+, avoiding redundancies between TUAB and AutoTUAB
            TUAB_basenames = [os.path.basename(fn) for fn in all_file_names]
            new_TUEG_inds = [i for i,fn in enumerate(all_labelled_TUEG_file_names) if os.path.basename(fn) not in TUAB_basenames]
            all_file_names += [all_labelled_TUEG_file_names[i] for i in new_TUEG_inds]
            labels += [TUEG_labels[i] for i in new_TUEG_inds]
        else:
            labels = TUEG_labels
            all_file_names = all_labelled_TUEG_file_names

    labels = np.array(labels).astype(np.int64)

    log.info("{:d} files in total.".format(len(all_file_names)))

    return all_file_names, labels


class DiagnosisSet(object):
    def __init__(self, n_recordings, max_recording_mins, preproc_functions,
                 data_folders,
                 train_or_eval='train', sensor_types=['EEG'], training_labels_from_csv=False,
                 balance_data=False, append_csv_set_to_TUAB=False):
        self.n_recordings = n_recordings
        self.max_recording_mins = max_recording_mins
        self.preproc_functions = preproc_functions
        self.train_or_eval = train_or_eval
        self.sensor_types = sensor_types
        self.data_folders = data_folders
        self.training_labels_from_csv = training_labels_from_csv
        self.balance_data = balance_data
        self.append_csv_set_to_TUAB = append_csv_set_to_TUAB

    def load(self, only_return_labels=False):
        log.info("Read file names")
        all_file_names, labels = get_all_sorted_file_names_and_labels(
            train_or_eval=self.train_or_eval,
            folders=self.data_folders,
            training_labels_from_csv=self.training_labels_from_csv,
            append_csv_set_to_TUAB=self.append_csv_set_to_TUAB)

        if self.max_recording_mins is not None:
            log.info("Read recording lengths...")
            assert 'train' == self.train_or_eval
            # Computation as:
            lengths = [get_recording_length(fname) for fname in all_file_names]
            lengths = np.array(lengths)
            mask = lengths < self.max_recording_mins * 60
            cleaned_file_names = np.array(all_file_names)[mask]
            cleaned_labels = labels[mask]
            lengths = lengths[mask]
            mask = lengths > 120
            cleaned_file_names = np.array(cleaned_file_names)[mask]
            cleaned_labels = cleaned_labels[mask]
        else:
            cleaned_file_names = np.array(all_file_names)
            cleaned_labels = labels

        if self.balance_data:
            # Whichever class we have more of, choose a random selection of the same size as the smaller class
            cleaned_file_names, cleaned_labels = self.__balance_data(cleaned_file_names, cleaned_labels)

        if only_return_labels:
            return cleaned_labels
        X = []
        y = []
        n_files = len(cleaned_file_names[:self.n_recordings])
        for i_fname, fname in enumerate(cleaned_file_names[:self.n_recordings]):
            if i_fname % 100 == 0:
                log.info("Load {:d} of {:d}".format(i_fname + 1,n_files))
            x = load_data(fname, preproc_functions=self.preproc_functions,
                          sensor_types=self.sensor_types)
            if x is not None and x.shape[1]>=6000:
                X.append(x)
                y.append(cleaned_labels[i_fname])
        y = np.array(y)
        return X, y

    def __balance_data(self, file_names, labels):
        n_abnormal = np.count_nonzero(labels == 1)
        n_normal = np.count_nonzero(labels == 0)
        log.info(f"Started with {n_abnormal} abnormal and {n_normal} normal recordings.")

        # Separate normals and abnormals
        abnormal_mask = labels == 1
        normal_mask = labels == 0
        abnormal_file_names = file_names[abnormal_mask]
        normal_file_names = file_names[normal_mask]

        # Reduce the larger class
        if n_abnormal > n_normal:
            log.info(f"Reducing # of abnormals from {n_abnormal} to {n_normal}")
            abnormal_file_names = self.__pick_n_by_subject(abnormal_file_names, n_normal)
            n_abnormal = n_normal
        else:
            print(f"Reducing # of normals from {n_normal} to {n_abnormal}")
            normal_file_names = self.__pick_n_by_subject(normal_file_names, n_abnormal)
            n_normal = n_abnormal

        # Recombine the data
        labels = np.concatenate((np.ones(n_abnormal, dtype=np.int64), np.zeros(n_normal, dtype=np.int64)))
        file_names = np.concatenate((abnormal_file_names, normal_file_names))

        # Shuffle normals and abnormals together
        shuffle_ind = np.random.permutation(n_normal+n_abnormal)
        labels = labels[shuffle_ind]
        file_names = file_names[shuffle_ind]
        return file_names, labels

    def __pick_n_by_subject(self, file_names, n):
        # Choose n files from file_names, maximising the spread across unique subjects:

        n_file_names = len(file_names)
        fns = np.copy(file_names)
        np.random.shuffle(fns)
        fns_to_pick_from = fns.tolist() # For ease of 'popping'
        random.shuffle(fns_to_pick_from) # Shuffle so that we don't always pick the first file in the session.
        fns_picked = []
        sub_IDs = [os.path.basename(fn).split('_')[0] for fn in file_names]
        unique_sub_IDs = np.unique(sub_IDs)
        n_subs = len(unique_sub_IDs)
        print(f"Picking {n} files from {n_subs} unique subjects.")
        k = 0
        for c in range(n):
            found = False
            while not found:
                sub_ID = unique_sub_IDs[k % n_subs]
                for i,s in enumerate(fns_to_pick_from):
                    if sub_ID in s:
                        fns_picked.append(fns_to_pick_from.pop(i))
                        found = True
                        break
                if not found:
                    # Try the next sub_ID
                    k += 1
                    assert(k <= n_file_names)

        return np.array(fns_picked)



