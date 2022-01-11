import logging
import time
from copy import copy
import sys

from config_overrider import new_config_values
import numpy as np
from numpy.random import RandomState
import resampy
import datetime #Imported for generating Log File Name
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn

th.cuda.empty_cache()

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity

from dataset import DiagnosisSet
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

datetime_object = datetime.datetime.now()
log_file_name_main = "Auto Diagnosis Log File " + datetime_object.strftime("%Y-%m-%d %H_%M_%S") + ".txt"

global X, y, test_X, test_y
X = y = test_X = test_y = None
reload_data_each_exp = True

def create_set(X, y, inds):
    """
    X list and y nparray
    :return: 
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y,):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.intersect1d(train_inds, test_inds).size == 0
        assert np.intersect1d(valid_inds, test_inds).size == 0
        assert np.array_equal(np.sort(
            np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set


class TrainValidSplitter(object):
    def __init__(self, n_folds, i_valid_fold, shuffle):
        self.n_folds = n_folds
        self.i_valid_fold = i_valid_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        valid_inds = folds[self.i_valid_fold]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, valid_inds)
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.array_equal(np.sort(np.union1d(train_inds, valid_inds)),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        return train_set, valid_set

def generate_overwritten_config(config, indicies, values):
    
    return_config = [""] * len(indicies)
    
    return_config[0] = CheckConfigValue(config.data_folders, values[0], indicies[0])
    return_config[1] = CheckConfigValue(config.n_recordings, values[1], indicies[1])
    return_config[2] = CheckConfigValue(config.sensor_types, values[2], indicies[2])
    return_config[3] = CheckConfigValue(config.n_chans, values[3], indicies[3])
    return_config[4] = CheckConfigValue(config.max_recording_mins, values[4], indicies[4])
    return_config[5] = CheckConfigValue(config.sec_to_cut, values[5], indicies[5])
    return_config[6] = CheckConfigValue(config.duration_recording_mins, values[6], indicies[6])
    return_config[7] = CheckConfigValue(config.test_recording_mins, values[7], indicies[7])
    return_config[8] = CheckConfigValue(config.max_abs_val, values[8], indicies[8])
    return_config[9] = CheckConfigValue(config.sampling_freq, values[9], indicies[9])
    return_config[10] = CheckConfigValue(config.divisor, values[10], indicies[10])
    return_config[11] = CheckConfigValue(config.test_on_eval, values[11], indicies[11])
    return_config[12] = CheckConfigValue(config.n_folds, values[12], indicies[12])
    return_config[13] = CheckConfigValue(config.i_test_fold, values[13], indicies[13])
    return_config[14] = CheckConfigValue(config.shuffle, values[14], indicies[14])
    return_config[15] = CheckConfigValue(config.model_name, values[15], indicies[15])
    return_config[16] = CheckConfigValue(config.n_start_chans, values[16], indicies[16])
    return_config[17] = CheckConfigValue(config.n_chan_factor, values[17], indicies[17])
    return_config[18] = CheckConfigValue(config.input_time_length, values[18], indicies[18])
    return_config[19] = CheckConfigValue(config.final_conv_length, values[19], indicies[19])
    return_config[20] = CheckConfigValue(config.model_constraint, values[20], indicies[20])
    return_config[21] = CheckConfigValue(config.init_lr, values[21], indicies[21])
    return_config[22] = CheckConfigValue(config.batch_size, values[22], indicies[22])
    return_config[23] = CheckConfigValue(config.max_epochs, values[23], indicies[23])
    return_config[24] = CheckConfigValue(config.cuda, values[24], indicies[24])
    return_config[25] = CheckConfigValue(config.training_labels_from_csv, values[25], indicies[25])
    return_config[26] = CheckConfigValue(config.append_csv_set_to_TUAB, values[26], indicies[26])


    return return_config
    
def CheckConfigValue(OriginalValue, OverwriteValue, OverwriteCheck):
    
    if OverwriteCheck == True:
        return OverwriteValue
    
    return OriginalValue

def run_exp(data_folders,
            n_recordings,
            sensor_types,
            n_chans,
            max_recording_mins,
            sec_to_cut, duration_recording_mins,
            test_recording_mins,
            max_abs_val,
            sampling_freq,
            divisor,
            test_on_eval,
            n_folds, i_test_fold,
            shuffle,
            model_name,
            n_start_chans, n_chan_factor,
            input_time_length, final_conv_length,
            model_constraint,
            init_lr,
            batch_size, max_epochs,cuda,
            training_labels_from_csv,
            append_csv_set_to_TUAB,
            config_index):
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(duration_recording_mins * 60 * fs)], fs))
    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))

    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))

    dataset = DiagnosisSet(n_recordings=n_recordings,
                           max_recording_mins=max_recording_mins,
                           preproc_functions=preproc_functions,
                           data_folders=data_folders,
                           train_or_eval='train',
                           sensor_types=sensor_types,
                           training_labels_from_csv=training_labels_from_csv,
                           append_csv_set_to_TUAB=append_csv_set_to_TUAB,
                           balance_data=training_labels_from_csv)
    if test_on_eval:
        if test_recording_mins is None:
            test_recording_mins = duration_recording_mins
        test_preproc_functions = copy(preproc_functions)
        test_preproc_functions[1] = lambda data, fs: (
            data[:, :int(test_recording_mins * 60 * fs)], fs)
        test_dataset = DiagnosisSet(n_recordings=n_recordings,
                                max_recording_mins=None,
                                preproc_functions=test_preproc_functions,
                                data_folders=data_folders,
                                train_or_eval='eval',
                                sensor_types=sensor_types)

    global X, y, test_X, test_y
    # if X is None:
    #     import pickle
    #     X, y = pickle.load( open( "G:/auto-eeg_AutoTUAB.pkl", "rb" ) ) # Caution: This will override paths specified in config_overrider.
    if reload_data_each_exp or X is None:
        X,y = dataset.load()
        import pickle
        pickle.dump([X, y], open( "G:/auto-eeg.pkl", "wb" ))

    # print(X)
    log.info(f"{len(X)} files in train+val data.")
    max_shape = np.max([list(x.shape) for x in X],
                       axis=0)
    assert max_shape[1] == int(duration_recording_mins *
                               sampling_freq * 60)
    if test_on_eval:
        if reload_data_each_exp or test_X is None:
            test_X, test_y = test_dataset.load()
        max_shape = np.max([list(x.shape) for x in test_X],
                           axis=0)
        assert max_shape[1] == int(test_recording_mins *
                                   sampling_freq * 60)
    if not test_on_eval:
        splitter = TrainValidTestSplitter(n_folds, i_test_fold,
                                          shuffle=shuffle)
        train_set, valid_set, test_set = splitter.split(X, y)
    else:
        splitter = TrainValidSplitter(n_folds, i_valid_fold=i_test_fold,
                                          shuffle=shuffle)
        train_set, valid_set = splitter.split(X, y)
        test_set = SignalAndTarget(test_X, test_y)
        #del test_X, test_y
    # del X,y # shouldn't be necessary, but just to make sure

    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    if model_name == 'shallow':
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_start_chans,
                                n_filters_spat=n_start_chans,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length).create_network()
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                         n_filters_time=n_start_chans,
                         n_filters_spat=n_start_chans,
                         input_time_length=input_time_length,
                         n_filters_2 = int(n_start_chans * n_chan_factor),
                         n_filters_3 = int(n_start_chans * (n_chan_factor ** 2.0)),
                         n_filters_4 = int(n_start_chans * (n_chan_factor ** 3.0)),
                         final_conv_length=final_conv_length,
                        stride_before_pool=True).create_network()
    elif (model_name == 'deep_smac'):
        if model_name == 'deep_smac':
            do_batch_norm = False
        else:
            assert model_name == 'deep_smac_bnorm'
            do_batch_norm = True
        double_time_convs = False
        drop_prob = 0.244445
        filter_length_2 = 12
        filter_length_3 = 14
        filter_length_4 = 12
        filter_time_length = 21
        final_conv_length = 1
        first_nonlin = elu
        first_pool_mode = 'mean'
        first_pool_nonlin = identity
        later_nonlin = elu
        later_pool_mode = 'mean'
        later_pool_nonlin = identity
        n_filters_factor = 1.679066
        n_filters_start = 32
        pool_time_length = 1
        pool_time_stride = 2
        split_first_layer = True
        n_chan_factor = n_filters_factor
        n_start_chans = n_filters_start
        model = Deep4Net(n_chans, n_classes,
                 n_filters_time=n_start_chans,
                 n_filters_spat=n_start_chans,
                 input_time_length=input_time_length,
                 n_filters_2=int(n_start_chans * n_chan_factor),
                 n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
                 n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
                 final_conv_length=final_conv_length,
                 batch_norm=do_batch_norm,
                 double_time_convs=double_time_convs,
                 drop_prob=drop_prob,
                 filter_length_2=filter_length_2,
                 filter_length_3=filter_length_3,
                 filter_length_4=filter_length_4,
                 filter_time_length=filter_time_length,
                 first_nonlin=first_nonlin,
                 first_pool_mode=first_pool_mode,
                 first_pool_nonlin=first_pool_nonlin,
                 later_nonlin=later_nonlin,
                 later_pool_mode=later_pool_mode,
                 later_pool_nonlin=later_pool_nonlin,
                 pool_time_length=pool_time_length,
                 pool_time_stride=pool_time_stride,
                 split_first_layer=split_first_layer,
                 stride_before_pool=True).create_network()
    elif model_name == 'shallow_smac':
        conv_nonlin = identity
        do_batch_norm = True
        drop_prob = 0.328794
        filter_time_length = 56
        final_conv_length = 22
        n_filters_spat = 73
        n_filters_time = 24
        pool_mode = 'max'
        pool_nonlin = identity
        pool_time_length = 84
        pool_time_stride = 3
        split_first_layer = True
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                n_filters_time=n_filters_time,
                                n_filters_spat=n_filters_spat,
                                input_time_length=input_time_length,
                                final_conv_length=final_conv_length,
                                conv_nonlin=conv_nonlin,
                                batch_norm=do_batch_norm,
                                drop_prob=drop_prob,
                                filter_time_length=filter_time_length,
                                pool_mode=pool_mode,
                                pool_nonlin=pool_nonlin,
                                pool_time_length=pool_time_length,
                                pool_time_stride=pool_time_stride,
                                split_first_layer=split_first_layer,
                                ).create_network()
    elif model_name == 'linear':
        model = nn.Sequential()
        model.add_module("conv_classifier",
                         nn.Conv2d(n_chans, n_classes, (600,1)))
        model.add_module('softmax', nn.LogSoftmax())
        model.add_module('squeeze', Expression(lambda x: x.squeeze(3)))
    else:
        assert False, "unknown model name {:s}".format(model_name)
    to_dense_prediction_model(model)
    log.info("Model:\n{:s}".format(str(model)))
    if cuda:
        model.cuda()
    # determine output size
    test_input = np_to_var(
        np.ones((2, n_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    log.info("In shape: {:s}".format(str(test_input.cpu().data.numpy().shape)))

    out = model(test_input)
    log.info("Out shape: {:s}".format(str(out.cpu().data.numpy().shape)))
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    log.info("{:d} predictions per input/trial".format(n_preds_per_input))
    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    if model_constraint is not None:
        assert model_constraint == 'defaultnorm'
        model_constraint = MaxNormDefaultConstraint()
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedDiagnosisMonitor(input_time_length, n_preds_per_input),
                RuntimeMonitor(),]
    stop_criterion = MaxEpochs(max_epochs)
    batch_modifier = None
    run_after_early_stop = True
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=run_after_early_stop,
                     batch_modifier=batch_modifier,
                     cuda=cuda)
    exp.run(log_file_name_main, config_index)
    return exp



if __name__ == "__main__":
    import config
    start_time = time.time()
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)

    config_var_length = 27

    for config_index in range(len(new_config_values)):

        modification_indicies = [False if new_config_values[config_index][k]==0 else True for k in range(config_var_length)]

        new_config = generate_overwritten_config(config, modification_indicies, new_config_values[config_index])
        
        exp = run_exp(
            new_config[0], #data_folders
            new_config[1], #n_recordings
            new_config[2], #sensor_types
            new_config[3], #n_chans
            new_config[4], #max_recording_mins
            new_config[5], #sec_to_cut
            new_config[6], #duration_recording_mins
            new_config[7], #test_recording_mins
            new_config[8], #max_abs_val
            new_config[9], #sampling_freq
            new_config[10], #divisor
            new_config[11], #test_on_eval
            new_config[12], #n_folds
            new_config[13], #i_test_fold
            new_config[14], #shuffle
            new_config[15], #model_name
            new_config[16], #n_start_chans
            new_config[17], #n_chan_factor
            new_config[18], #input_time_length
            new_config[19], #final_conv_length
            new_config[20], #model_constraint
            new_config[21], #init_lr
            new_config[22], #batch_size
            new_config[23], #max_epochs
            new_config[24], #cuda
            new_config[25], #training_labels_from_csv
            new_config[26], #append_csv_set_to_TUAB
            config_index + 1)   #config_index, +1 as starts at 0              
        end_time = time.time()
        run_time = end_time - start_time
    
        log.info("Experiment runtime: {:.2f} sec".format(run_time))
    
        # In case you want to recompute predictions for further analysis:
        exp.model.eval()
        for setname in ('train', 'valid', 'test'):
            log.info("Compute predictions for {:s}...".format(
                setname))
            dataset = exp.datasets[setname]
            batches = exp.iterator.get_batches(dataset, shuffle=False)
            if config.cuda:
                preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                          for b in batches]
            else:
                preds_per_batch = [var_to_np(exp.model(np_to_var(b[0])))
                          for b in batches]

            # for index in sorted(to_remove, reverse=True):
            #     del dataset.X[index]
            #     try:
            #         del dataset.Y[index]
            #     except:
            #         pass
            preds_per_trial = compute_preds_per_trial(
                preds_per_batch, dataset,
                input_time_length=exp.iterator.input_time_length,
                n_stride=exp.iterator.n_preds_per_input)
            mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                        preds_per_trial]
            mean_preds_per_trial = np.array(mean_preds_per_trial)
    

            
        
    
    