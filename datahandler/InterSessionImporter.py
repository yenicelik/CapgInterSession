from __future__ import print_function

import os
import sys
import scipy.io as sio
import numpy as np
from config import *

import logging
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging = logging.getLogger(__name__)

class InterSessionImporter(object):

    def __init__(self, data_parent_dir="datahandler/Datasets/Preprocessed/DB-a"):
        filepaths = self.get_filepaths_in_directory(data_parent_dir)
        data = self.get_data_from_filepaths(filepaths)

        X, y, sids = self.get_train_cv_test_given_sid(data)
        X = np.reshape(X, (-1, 16, 8))

        self.X = X
        self.y = y
        self.sids = sids

    #TODO: check in the paper how training data is really provided to the algorithm
    def get_train_cv_test_given_sid(self, sid, cv_size=0.4, shuffle=True):
        assert sid < 18, "There are only 18 possible subjects to choose from! :: {}".format(sid)
        cv_split = self.X.shape[0] * cv_size
        train_indices = (self.sids != sid)
        test_indices = (self.sids == sid)

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        #Get training set
        X_train = self.X[train_indices]
        y_train = self.y[train_indices]
        sid_train = self.sids[train_indices]

        #Get cv and test sets
        X_cv = self.X[train_indices[:cv_split]]
        y_cv = self.y[train_indices[:cv_split]]
        sid_cv = self.sids[train_indices[:cv_split]]

        X_test = self.X[train_indices[cv_split:]]
        y_test = self.y[train_indices[cv_split:]]
        sid_test = self.sids[train_indices[cv_split:]]

        return X_train, y_train, sid_train, X_cv, y_cv, sid_cv, X_test, y_test, sid_test


    def get_train_cv_test(self, shuffle=True):
        ##TRAINING SET
        X_train = np.reshape(self.X_odd, (-1, 1000, 16, 8))
        y_train = np.reshape(self.y_odd, (-1, 1000, NUM_GESTURES))
        n = X_train.shape[0]

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        X_train = X_train[indices]
        X_train = np.reshape(X_train, (-1, 16, 8))
        y_train = y_train[indices]
        y_train = np.reshape(y_train, (-1, NUM_GESTURES))


        #CV AND TEST SET
        X_dev = np.reshape(self.X_even, (-1, 1000, 16, 8))
        y_dev = np.reshape(self.y_even, (-1, 1000, NUM_GESTURES))
        n = X_dev.shape[0]

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        test_ind, cv_ind = np.split(indices, 2)

        X_cv = X_dev[cv_ind,:]
        X_cv = np.reshape(X_cv, (-1, 16, 8))
        y_cv = y_dev[cv_ind,:]
        y_cv = np.reshape(y_cv, (-1, NUM_GESTURES))

        X_test = X_dev[test_ind,:]
        X_test = np.reshape(X_test, (-1, 16, 8))
        y_test = y_dev[test_ind,:]
        y_test = np.reshape(y_test, (-1, NUM_GESTURES))

        return X_train, y_train, X_cv, y_cv, X_test, y_test



    def get_filepaths_in_directory(self, parent_directory):
        out = []
        for path, _, files in os.walk(parent_directory):
            for filename in files:
                if filename.endswith(".mat"): #this is for the specific dataset of CPMyo
                    logging.debug("Found file {}".format(filename))
                    filepath = os.path.join(path, filename)
                    out.append(filepath)
        return out


    def get_data_from_filepaths(self, filepaths):
        if len(filepaths) == 0:
            logging.error("No filepaths are given! Quitting from get_data_from_filepath")
            logging.error(os.getcwd())
            sys.exit(69)
        if type(filepaths) != type([1, 2]):
            logging.error("Filepaths is not an array!")
            sys.exit(69)

        out = []
        for filepath in filepaths:
            filedata = sio.loadmat(filepath)
            out.append(filedata)

        return out


    def get_X_y_sid_from_data(self, data_arr):
        ys = []
        Xs = []
        sids = []

        for ele in data_arr:
            X = ele['data']
            #Turning into one-hot
            gesture = ele['gesture']
            sid = ele['subject']
            if gesture == 101 and NUM_GESTURES == 10:
                gesture = 10
            elif gesture == 100 and NUM_GESTURES == 10:
                gesture = 9
            elif gesture == 100 or gesture == 101:
                continue
            y = np.zeros((X.shape[0], NUM_GESTURES)) #10 dimensions in one-hot setting
            y[:,gesture-1] = 1
            #splitting array now again, feeling a little unconfident here, as off-by-one error are critical
            ys.append(y)
            Xs.append(X)
            sids.append(sid)

        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(sids, axis=0)
