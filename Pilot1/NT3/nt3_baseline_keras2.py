from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import tensorflow as tf
import horovod.tensorflow as hvd
from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

TIMEOUT = -1 # in sec; set this to -1 for no timeout 
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import data_utils
import p1_common, p1_common_keras
from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut

import horovod.tensorflow as hvd
# Initialize hovorod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


#url_nt3 = 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/normal-tumor/'
#file_train = 'nt_train2.csv'
#file_test = 'nt_test2.csv'

#EPOCH = 400
#BATCH = 20
#CLASSES = 2

#PL = 60484   # 1 + 60483 these are the width of the RNAseq datasets
#P     = 60483   # 60483
#DR    = 0.1      # Dropout rate

def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'nt3_default_model.txt'),
                        help="specify model configuration file")

    # Parse has been split between arguments that are common with the default neon parser
    # and all the other options
    parser = p1_common.get_default_neon_parse(parser)
    parser = p1_common.get_p1_common_parser(parser)

    return parser

def get_nt3_parser():

	parser = argparse.ArgumentParser(prog='nt3_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Train Autoencoder - Pilot 1 Benchmark NT3')

	return common_parser(parser)

def read_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    section = config.sections()
    fileParams = {}

    fileParams['data_url'] = eval(config.get(section[0],'data_url'))
    fileParams['train_data'] = eval(config.get(section[0],'train_data'))
    fileParams['test_data'] = eval(config.get(section[0],'test_data'))
    fileParams['model_name'] = eval(config.get(section[0],'model_name'))
    fileParams['conv'] = eval(config.get(section[0],'conv'))
    fileParams['dense'] = eval(config.get(section[0],'dense'))
    fileParams['activation'] = eval(config.get(section[0],'activation'))
    fileParams['out_act'] = eval(config.get(section[0],'out_act'))
    fileParams['loss'] = eval(config.get(section[0],'loss'))
    fileParams['optimizer'] = eval(config.get(section[0],'optimizer'))
    fileParams['metrics'] = eval(config.get(section[0],'metrics'))
    fileParams['epochs'] = eval(config.get(section[0],'epochs'))
    fileParams['batch_size'] = eval(config.get(section[0],'batch_size'))
    fileParams['learning_rate'] = eval(config.get(section[0], 'learning_rate'))
    fileParams['drop'] = eval(config.get(section[0],'drop'))
    fileParams['classes'] = eval(config.get(section[0],'classes'))
    fileParams['pool'] = eval(config.get(section[0],'pool'))
    fileParams['save'] = eval(config.get(section[0], 'save'))

    # parse the remaining values
    for k,v in config.items(section[0]):
        if not k in fileParams:
            fileParams[k] = eval(v)

    return fileParams

def initialize_parameters():
    # Get command-line parameters
    parser = get_nt3_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    return gParameters


def load_data(train_path, test_path, gParameters):

    print('Loading data...')
    df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train,gParameters['classes'])
    Y_test = np_utils.to_categorical(df_y_test,gParameters['classes'])

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    my_mpi_rank = hvd.rank()
    total_ranks = hvd.size()
    input_size = df_train.shape[0]
    per_rank_count =  int( input_size / total_ranks)
    overloaded_ranks = input_size % total_ranks
    if my_mpi_rank < overloaded_ranks: 
        start = my_mpi_rank * (per_rank_count + 1) 
        stop = start + (per_rank_count)
    else:
        start = my_mpi_rank * per_rank_count + overloaded_ranks 
        stop = start + per_rank_count - 1
        
    print ("size:{0}, rank:{1}, start:{2}, stop:{3}".format(total_ranks, my_mpi_rank, start, stop))

#        X_train = df_x_train.as_matrix()
#        X_test = df_x_test.as_matrix()

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return X_train[start:stop, :], Y_train[start:stop, :], X_test, Y_test


def run(gParameters):

    print ('Params:', gParameters)

    file_train = gParameters['train_data']
    file_test = gParameters['test_data']
    url = gParameters['data_url']

    train_file = data_utils.get_file(file_train, url+file_train, cache_subdir='Pilot1')
    test_file = data_utils.get_file(file_test, url+file_test, cache_subdir='Pilot1')

    X_train, Y_train, X_test, Y_test = load_data(train_file, test_file, gParameters)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    x_train_len = X_train.shape[1]

    # this reshaping is critical for the Conv1D to work

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    model = Sequential()

    layer_list = list(range(0, len(gParameters['conv']), 3))
    for l, i in enumerate(layer_list):
        filters = gParameters['conv'][i]
        filter_len = gParameters['conv'][i+1]
        stride = gParameters['conv'][i+2]
        print(int(i/3), filters, filter_len, stride)
        if gParameters['pool']:
            pool_list=gParameters['pool']
            if type(pool_list) != list:
                pool_list=list(pool_list)

        if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
        if 'locally_connected' in gParameters:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
        else:
            #input layer
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid'))
        model.add(Activation(gParameters['activation']))
        if gParameters['pool']:
                model.add(MaxPooling1D(pool_size=pool_list[int(i/3)]))

    model.add(Flatten())

    for layer in gParameters['dense']:
        if layer:
            model.add(Dense(layer))
            model.add(Activation(gParameters['activation']))
            if gParameters['drop']:
                    model.add(Dropout(gParameters['drop']))
    model.add(Dense(gParameters['classes']))
    model.add(Activation(gParameters['out_act']))

#Reference case
#model.add(Conv1D(filters=128, kernel_size=20, strides=1, padding='valid', input_shape=(P, 1)))
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=1))
#model.add(Conv1D(filters=128, kernel_size=10, strides=1, padding='valid'))
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=10))
#model.add(Flatten())
#model.add(Dense(200))
#model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(Dense(20))
#model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(Dense(CLASSES))
#model.add(Activation('softmax'))

    kerasDefaults = p1_common.keras_default_config()

    # Define optimizer
    # optimizer = p1_common_keras.build_optimizer(gParameters['optimizer'],
    #                                             gParameters['learning_rate'],
    #                                             kerasDefaults)

    # Adjust learning rate based on number of GPUs (naive approach).
    opt = tf.train.GradientDescentOptimizer(gParameters['learning_rate'] * hvd.size())

    # Add Horovod Distributed Optimizer.
    optimizer = hvd.DistributedOptimizer(opt)

    model.summary()
    model.compile(loss=gParameters['loss'],
                  optimizer=optimizer,
                  metrics=[gParameters['metrics']])

    output_dir = gParameters['save']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # calculate trainable and non-trainable params
    gParameters.update(compute_trainable_params(model))

    # set up a bunch of callbacks to do work during model training..
    model_name = gParameters['model_name']
    path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
    # checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('{}/training.log'.format(output_dir))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    candleRemoteMonitor = CandleRemoteMonitor(params=gParameters)
    timeoutMonitor = TerminateOnTimeOut(TIMEOUT)
    history = model.fit(X_train, Y_train,
                    batch_size=gParameters['batch_size'],
                    epochs=gParameters['epochs'],
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks = [csv_logger, candleRemoteMonitor, timeoutMonitor])

    score = model.evaluate(X_test, Y_test, verbose=0)

    if True:
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        # serialize model to JSON
        model_json = model.to_json()
        with open("{}/{}.model.json".format(output_dir, model_name), "w") as json_file:
            json_file.write(model_json)

        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open("{}/{}.model.yaml".format(output_dir, model_name), "w") as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        model.save_weights("{}/{}.weights.h5".format(output_dir, model_name))
        print("Saved model to disk")

        # load json and create model
        json_file = open('{}/{}.model.json'.format(output_dir, model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model_json = model_from_json(loaded_model_json)


        # load yaml and create model
        yaml_file = open('{}/{}.model.yaml'.format(output_dir, model_name), 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model_yaml = model_from_yaml(loaded_model_yaml)


        # load weights into new model
        loaded_model_json.load_weights('{}/{}.weights.h5'.format(output_dir, model_name))
        print("Loaded json model from disk")

        # evaluate json loaded model on test data
        loaded_model_json.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
        score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

        print('json Test score:', score_json[0])
        print('json Test accuracy:', score_json[1])

        print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))

        # load weights into new model
        loaded_model_yaml.load_weights('{}/{}.weights.h5'.format(output_dir, model_name))
        print("Loaded yaml model from disk")

        # evaluate loaded model on test data
        loaded_model_yaml.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
        score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

        print('yaml Test score:', score_yaml[0])
        print('yaml Test accuracy:', score_yaml[1])

        print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1]*100))

    return history

def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
