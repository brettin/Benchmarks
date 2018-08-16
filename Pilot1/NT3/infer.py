from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import time
import gzip
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

TIMEOUT=3600 # in sec; set this to -1 for no timeoutA
 
# needed to find libs in ../common and ../../common
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import data_utils
import p1_common, p1_common_keras
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from sklearn.metrics import accuracy_score

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
    description = 'Perform inference for normal-tumor classification.'
    parser = argparse.ArgumentParser(prog='infer', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=description)
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
    fileParams['output_dir'] = eval(config.get(section[0], 'output_dir'))

    # parse the remaining values
    for k,v in config.items(section[0]):
        if not k in fileParams:
            fileParams[k] = eval(v)

    return fileParams

def initialize_parameters():
    parser = get_nt3_parser()
    args = parser.parse_args()
    fileParameters = read_config_file(args.config_file)
    print ('Params:', fileParameters)
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    return gParameters

gParameters = initialize_parameters()
print('learning_rate is ', gParameters['learning_rate'])
print('model_name is ', gParameters['model_name'])
print('output_dir is ', gParameters['output_dir'])

# todo: enable user to specify csv file to load.
def load_data(train_path, test_path, gParameters):
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')

    print('df_test shape:', df_test.shape)

    seqlen = df_test.shape[1]

    df_y_test = df_test[:,0].astype('int')

    Y_test = np_utils.to_categorical(df_y_test,gParameters['classes'])

    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = scaler.fit_transform(X_test)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return X_train, Y_train, X_test, Y_test

# load json and create model
json_file = open('{}/{}.model.json'.format(gParameters['output_dir'], gParameters['model_name']), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)

# load weights into new model
loaded_model_json.load_weights('{}/{}.weights.h5'.format(gParameters['output_dir'], gParameters['model_name']))

# loaad test and train data as samples to infer ond
file_train = gParameters['train_data']
file_test = gParameters['test_data']
url = gParameters['data_url']

train_file = data_utils.get_file(file_train, url+file_train, cache_subdir='Pilot1')
test_file = data_utils.get_file(file_test, url+file_test, cache_subdir='Pilot1')
X_train, Y_train, X_test, Y_test = load_data(train_file, test_file, gParameters)
print ('X_test shape: ', X_test.shape)

# this reshaping is critical for the Conv1D to work
X_test = np.expand_dims(X_test, axis=2)
print('X_test shape" ', X_test.shape)

# shuffle a column if looking at feature importance
fileprefix = 'out'
if 'shuffle_col' in gParameters:
    print(str(datetime.now()),  " shuffling column ", gParameters['shuffle_col'])
    X_test[gParameters['shuffle_col']] = np.random.permutation(X_test[gParameters['shuffle_col']])
    fileprefix = fileprefix + '.' + str(gParameters['shuffle_col'])

# do prediction
loaded_model_json.compile(loss=gParameters['loss'],
    optimizer=gParameters['optimizer'],
    metrics=[gParameters['metrics']])
prediction = loaded_model_json.predict(X_test, verbose=0)
print( prediction )

index = np.array([])
for row in prediction:
    print (np.argmax(row))
    index = np.append(index, [np.argmax(row)])

np.savetxt(fileprefix, index, '%d')
print (str(datetime.now()),  " done")
